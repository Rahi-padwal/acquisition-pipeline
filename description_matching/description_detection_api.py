"""Detect acquisitions from a company's Elasticsearch `description` field.

This module is separate from the existing companyName-based acquisition
detection elsewhere in this project (see check_name_signals.py) - it only
looks at the free-text `description` field for acquisition-indicating
phrases ("was acquired by X", "operates as a wholly-owned subsidiary of X",
etc.) and extracts a candidate acquirer name from each match.

Standalone: has its own FastAPI app, its own Elasticsearch client, and its
own .env loading. Not wired into api.py / pipeline.py - that's a follow-up.

Run with: python -m uvicorn description_detection_api:app --reload --port 8003
"""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")
MAX_WORKERS = int(os.environ.get("DESCRIPTION_DETECTION_MAX_WORKERS", "15"))

# Cluster intermittently 404s transiently - retry before giving up.
_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


# --- Pattern definitions -----------------------------------------------------

# alnum start, up to ~50 name-shaped chars, non-greedy, stopping at a
# sentence-ending punctuation mark or end of string.
_NAME_CAPTURE = r"[A-Za-z0-9][^,.;:()\n]{0,49}?"
_NAME_TERMINATOR = r"(?=[,.;:)]|$)"

_WAS_ACQUIRED_BY = re.compile(
    r"was\s+acquired\s+by\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_WHOLLY_OWNED_SUBSIDIARY = re.compile(
    r"operates\s+as\s+an?\s+wholly[\s-]+owned\s+subsidiary\s+of\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_JOINED_FORCES = re.compile(
    r"has\s+joined\s+forces\s+with\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_FORMERLY = re.compile(
    r"\bformerly\s+(?:known\s+as\s+)?(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_STAY_UP_TO_DATE = re.compile(
    r"to\s+stay\s+up\s+to\s+date\s+on\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
# "X is now Y" / "X are now Y" - primary, high-confidence trigger (same tier
# as was_acquired_by). Added because the real acquisition signal is often
# stated this way rather than via "formerly" - e.g. "Emburse Go (formerly
# Roadmap) is now Emburse." - the acquirer here is "Emburse", not "Roadmap".
_IS_NOW = re.compile(
    r"\b(?:is|are)\s+now\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)

# Bare "formerly" count, used by the consolidator heuristic below - doesn't
# need a capture group, just a count of occurrences.
_FORMERLY_WORD = re.compile(r"\bformerly\b", re.IGNORECASE)

# Lightweight guard (rule 4): does the description open by describing the
# SUBJECT as a parent/umbrella entity ("We are X...", "X is the leading...")?
# Checked only against the first ~100 chars - not a general subject-detector,
# just enough to reinforce the consolidator heuristic below.
_SUBJECT_POSITION = re.compile(
    r"^\s*(?:we\s+are\b|[A-Za-z][\w&.,'-]{0,60}?\s+is\s+the\s+(?:largest|leading|nation'?s|world'?s|premier|top))",
    re.IGNORECASE,
)

# One entry per phrase, in extraction priority order (see
# extract_acquirer_from_description for how this ordering is used). NOTE:
# "formerly" is NOT a primary trigger despite being listed here - see the
# comment on the formerly branch in extract_acquirer_from_description for why.
_DESCRIPTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("was_acquired_by", _WAS_ACQUIRED_BY),
    ("wholly_owned_subsidiary", _WHOLLY_OWNED_SUBSIDIARY),
    ("joined_forces", _JOINED_FORCES),
    ("is_now", _IS_NOW),
    ("formerly", _FORMERLY),
    ("stay_up_to_date", _STAY_UP_TO_DATE),
]


def _clean_acquirer(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip().strip(")").strip()


def extract_acquirer_from_description(description: str | None) -> dict[str, Any] | None:
    """Scan `description` for acquisition-indicating phrases.

    Tries patterns in priority order: was_acquired_by > wholly_owned_subsidiary
    > joined_forces > is_now > formerly (supporting-only, see below) >
    stay_up_to_date. Returns None if there's no description or nothing
    matches, otherwise:
        {"acquirerCompany": str | None, "matchedPattern": str,
         "confidence": "high" | "low", "nameChanged": bool,
         "isLikelyAcquirer": bool, "rawText": str}
    `rawText` is the matched phrase itself, for use as evidence.

    "formerly" is deliberately NOT a primary trigger. Real-world testing
    showed "formerly X" almost never means the SUBJECT was acquired - it
    only means a name change happened somewhere in the text, and just as
    often signals a self-rebrand ("Formerly known as PNM Group, SAHBIA Group
    is a global digital agency...") or the subject itself being the
    consolidator narrating ITS OWN acquired subsidiaries' old names ("We are
    NTT DATA Payment Services, comprising... (formerly GHL)... (formerly
    iPay88)..."). Do not re-promote "formerly" to a primary trigger without
    re-reading those false-positive cases.
    """
    text = (description or "").strip()
    if not text:
        return None

    formerly_count = len(_FORMERLY_WORD.findall(text))
    subject_is_parent = bool(_SUBJECT_POSITION.search(text[:100]))

    # Consolidator heuristic: 2+ "formerly" mentions in one description is a
    # much stronger signal that the subject is a serial acquirer describing
    # its own subsidiaries' old names than that the subject itself was
    # acquired - this is exactly the NTT DATA / Moore Response Management
    # Group false-positive pattern. A single mention counts too when the
    # opening clause already frames the subject as the parent/umbrella
    # entity (the subject-position guard). This overrides ANY primary-
    # pattern match found below, per spec - even a genuine "was acquired by"
    # elsewhere in the same text is suppressed, since scoping that
    # interaction correctly is out of scope for this fix.
    if formerly_count >= 2 or (formerly_count == 1 and subject_is_parent):
        formerly_match = _FORMERLY.search(text)
        return {
            "acquirerCompany": None,
            "matchedPattern": "formerly",
            "confidence": "low",
            "nameChanged": True,
            "isLikelyAcquirer": True,
            "rawText": formerly_match.group(0) if formerly_match else "formerly",
        }

    for pattern_name, pattern in (
        ("was_acquired_by", _WAS_ACQUIRED_BY),
        ("wholly_owned_subsidiary", _WHOLLY_OWNED_SUBSIDIARY),
        ("joined_forces", _JOINED_FORCES),
        ("is_now", _IS_NOW),
    ):
        m = pattern.search(text)
        if m:
            return {
                "acquirerCompany": _clean_acquirer(m.group("acquirer")),
                "matchedPattern": pattern_name,
                "confidence": "high",
                # formerly demoted: a same-text "formerly" mention no longer
                # drives its own match, it just rides along as evidence that
                # the name also changed alongside this stronger signal.
                "nameChanged": formerly_count > 0,
                "isLikelyAcquirer": False,
                "rawText": m.group(0),
            }

    formerly_match = _FORMERLY.search(text)
    if formerly_match:
        # Demoted: none of the stronger acquisition phrases matched above,
        # and the consolidator heuristic above didn't trigger either, so
        # this is a lone "formerly X" with no other corroborating signal.
        # Never returns an acquirerCompany / drives isAcquired=true on its
        # own - see the false-positive cases in this function's docstring.
        return {
            "acquirerCompany": None,
            "matchedPattern": "formerly",
            "confidence": "low",
            "nameChanged": True,
            "isLikelyAcquirer": False,
            "rawText": formerly_match.group(0),
        }

    stay_up_to_date_match = _STAY_UP_TO_DATE.search(text)
    if stay_up_to_date_match:
        return {
            "acquirerCompany": _clean_acquirer(stay_up_to_date_match.group("acquirer")),
            "matchedPattern": "stay_up_to_date",
            "confidence": "low",
            "nameChanged": False,
            "isLikelyAcquirer": False,
            "rawText": stay_up_to_date_match.group(0),
        }

    return None


# --- Elasticsearch lookup ----------------------------------------------------


def _get_document(dwh_id: str, es_client: Elasticsearch) -> dict[str, Any] | None:
    """Fetch {dwhId, companyName, description} by dwhId. Returns None if not
    found. Raises on ES/connection failure - caller decides how to handle.

    Retries on transient errors (the cluster intermittently 404s on a valid
    index/doc - see pipeline.py's _search_with_retry for the same pattern)."""
    body = {
        "query": {"term": {"dwhId": dwh_id}},
        "size": 1,
        "_source": ["dwhId", "companyName", "description"],
    }

    last_error: Exception | None = None
    resp = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            resp = es_client.search(index=ES_INDEX, body=body)
            break
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_SEARCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    if resp is None:
        raise last_error  # type: ignore[misc]

    hits = resp["hits"]["hits"]
    if not hits:
        return None
    return hits[0]["_source"]


def _not_acquired_result(dwh_id: str, company_name: str | None) -> dict[str, Any]:
    return {
        "company_id": dwh_id,
        "company_name": company_name,
        "is_acquired": False,
        "likely_acquirer": None,
        "evidence": [],
    }


def _build_result(dwh_id: str, source: dict[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return _not_acquired_result(dwh_id, None)

    company_name = source.get("companyName")
    match = extract_acquirer_from_description(source.get("description"))
    # A match with no acquirerCompany (demoted "formerly", or the
    # consolidator heuristic) is evidence of a name change, not an
    # acquisition of the subject - never surface it as is_acquired=true.
    if match is None or match["acquirerCompany"] is None:
        return _not_acquired_result(dwh_id, company_name)

    return {
        "company_id": dwh_id,
        "company_name": company_name,
        "is_acquired": True,
        "likely_acquirer": match["acquirerCompany"],
        "evidence": [
            {
                "candidate_name": match["acquirerCompany"],
                "matched_pattern": match["matchedPattern"],
                "raw_name": match["rawText"],
            }
        ],
    }


def fetch_and_detect(dwh_id: str, es_client: Elasticsearch) -> dict[str, Any]:
    """Fetch dwh_id's document and run description-based acquisition
    detection on it. Never raises - failures come back as is_acquired=None."""
    try:
        source = _get_document(dwh_id, es_client)
    except Exception:
        logger.exception("dwhId %s: failed to fetch from Elasticsearch", dwh_id)
        return {
            "company_id": dwh_id,
            "company_name": None,
            "is_acquired": None,
            "likely_acquirer": None,
            "evidence": [],
        }

    return _build_result(dwh_id, source)


# --- FastAPI app --------------------------------------------------------------

app = FastAPI(title="Description-Based Acquisition Detector")

_es_client = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)


class Evidence(BaseModel):
    candidate_name: str | None
    matched_pattern: str
    raw_name: str


class DetectionResult(BaseModel):
    company_id: str
    company_name: str | None
    is_acquired: bool | None
    likely_acquirer: str | None
    evidence: list[Evidence]


class BatchRequest(BaseModel):
    dwhIds: list[str]


@app.get("/detect-acquisition/{dwh_id}", response_model=DetectionResult)
async def detect_acquisition(dwh_id: str) -> DetectionResult:
    try:
        source = _get_document(dwh_id, _es_client)
    except Exception as exc:
        logger.exception("dwhId %s: Elasticsearch call failed", dwh_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if source is None:
        raise HTTPException(status_code=404, detail=f"No document found for dwhId={dwh_id}")

    return DetectionResult(**_build_result(dwh_id, source))


@app.post("/detect-acquisition/batch", response_model=list[DetectionResult])
async def detect_acquisition_batch(body: BatchRequest) -> list[DetectionResult]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_and_detect, dwh_id, _es_client): dwh_id for dwh_id in body.dwhIds}
        for future in as_completed(futures):
            dwh_id = futures[future]
            results[dwh_id] = future.result()

    # Preserve the order dwhIds were requested in.
    return [DetectionResult(**results[dwh_id]) for dwh_id in body.dwhIds]
