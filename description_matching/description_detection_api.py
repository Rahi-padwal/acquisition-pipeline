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
# --- joined_forces-specific list/partnership guard ---------------------------
#
# "has joined forces with X" is genuinely ambiguous in the wild: the same
# phrasing shows up both for real acquisitions ("has joined forces with Acme
# Corp") AND for partnership/client-roster announcements ("has joined forces
# with industry giants like Acme Corp", "has joined forces with Acme, Beta,
# and Gamma") - the latter names a PARTNER or CLIENT, not an acquirer, and
# should never be extracted as a confident single acquirer. Two independent,
# vocabulary-general signals catch this shape without hardcoding any specific
# company or noun list:
#   (1) a list-introducing connector ("like" / "such as" / "including" / ...)
#       sitting between the trigger phrase and the captured name - catches
#       "industry giants like X", "leading companies such as X", "clients
#       including X", etc. regardless of which noun precedes the connector.
#   (2) multiple comma- or "and"-joined capitalized names attached to the
#       captured name (either right after it, or "and"-joined inside it) -
#       catches "has joined forces with X, Y, and Z" even when there's no
#       connector word at all.
# Either signal firing means this match is discarded entirely rather than
# scored (see the joined_forces branch in extract_acquirer_from_description).
_JOINED_FORCES_LIST_CONNECTOR = re.compile(
    r"(?:like|such\s+as|including|comprising|e\.g\.?|to\s+name\s+(?:a\s+few|just\s+one|but\s+a\s+few))\s*$",
    re.IGNORECASE,
)
_JOINED_FORCES_MULTI_ENTITY_TAIL = re.compile(r"^(?:,\s*[A-Z][\w&.'-]*|\s+and\s+[A-Z])")
# Deliberately NOT re.IGNORECASE: "and" joining two separate entities inside
# the captured name (e.g. "Acme and Beta") is only a reliable list signal
# when "and" is lowercase and what follows is capitalized like a new proper
# noun - a real two-word company name containing "and" ("Johnson and
# Johnson"-style) can still slip through here, accepted as a rare tradeoff.
_JOINED_FORCES_MULTI_ENTITY_INTERNAL_AND = re.compile(r"[\w&.'-]+\s+and\s+[A-Z]")

# Filler words between "joined forces with" and the actual acquirer name
# ("industry giants like ", "leading companies such as ") are skipped
# generically - any run of lowercase-initial words - rather than captured, so
# the acquirer capture starts at the first proper-noun-like (capitalized)
# token instead of the first word after the trigger phrase. (?-i:...) turns
# case-folding off locally so this stays genuinely case-sensitive even though
# the rest of the pattern is compiled with re.IGNORECASE. Tradeoff: a real
# acquirer name stylized entirely in lowercase won't be captured by this
# pattern; accepted as rare relative to the false-positive rate this fixes.
_JOINED_FORCES_FILLER = r"(?:(?-i:[a-z][\w'-]*)\s+){0,4}"
_JOINED_FORCES_ACQUIRER_CAPTURE = r"(?-i:[A-Z])[^,.;:()\n]{0,49}?"

_JOINED_FORCES = re.compile(
    r"has\s+joined\s+forces\s+with\s+" + _JOINED_FORCES_FILLER
    + r"(?P<acquirer>" + _JOINED_FORCES_ACQUIRER_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)


def _is_joined_forces_list_announcement(text: str, match: re.Match) -> bool:
    """True when a _JOINED_FORCES match looks like a partnership/client-list
    announcement rather than a genuine single-acquirer acquisition phrase -
    see the module-level comment above _JOINED_FORCES_LIST_CONNECTOR for the
    two signals checked."""
    lead = match.group(0)[: match.start("acquirer") - match.start(0)]
    if _JOINED_FORCES_LIST_CONNECTOR.search(lead):
        return True

    acquirer_text = match.group("acquirer")
    if _JOINED_FORCES_MULTI_ENTITY_INTERNAL_AND.search(acquirer_text):
        return True

    tail = text[match.end("acquirer") : match.end("acquirer") + 60]
    return bool(_JOINED_FORCES_MULTI_ENTITY_TAIL.match(tail))


# --- joined_forces-specific non-acquirer / garbage-text filters --------------
#
# Manual review of ~40 real powered_by/joined_forces matches (name_matching
# side and this description side together) turned up two more recurring
# false-positive categories beyond the list-announcement shape above:
# franchise-network/accelerator-fund organizations getting read as an
# acquirer, and captures that are just over-grabbed prose with no real
# company name in them. Mirrors name_matching/check_name_signals.py's
# _is_non_acquirer_organization()/_is_garbage_acquirer_text() - duplicated
# rather than imported since this module is intentionally standalone (own
# FastAPI app, own ES client - see the module docstring), and only applied to
# joined_forces here (this module has no powered_by pattern of its own).
_NON_ACQUIRER_ORG_PREFIXES = ("proforma",)
_ACCELERATOR_FUND_RE = re.compile(r"\b(accelerators?|akcelerator|funds?|idea)\b", re.IGNORECASE)


def _is_non_acquirer_organization(candidate: str) -> bool:
    normalized = candidate.strip().lower()
    if any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in _NON_ACQUIRER_ORG_PREFIXES):
        return True
    return bool(_ACCELERATOR_FUND_RE.search(candidate))


_ACQUIRER_FILLER_PHRASE_RE = re.compile(
    r"\b(?:leading|major|other|various|several|top|well[\s-]known|renowned|trusted)\b|\b(?:like|such\s+as|including)\b",
    re.IGNORECASE,
)
_POWERED_BY_JOINED_FORCES_GENERIC_WORDS = {
    "ai", "growth", "application", "applications", "verification",
    "solution", "solutions", "platform", "platforms", "software",
    "technology", "technologies", "digital", "data", "analytics",
    "insights", "innovation", "innovations", "systems", "system",
    "service", "services", "cloud", "network", "networks",
}


def _is_garbage_acquirer_text(candidate: str) -> bool:
    if _ACQUIRER_FILLER_PHRASE_RE.search(candidate):
        return True
    capitalized_words = [w.strip(".,'-&") for w in candidate.split() if w[:1].isupper()]
    capitalized_words = [w for w in capitalized_words if w]
    if not capitalized_words:
        return True
    return all(w.lower() in _POWERED_BY_JOINED_FORCES_GENERIC_WORDS for w in capitalized_words)


_FORMERLY = re.compile(
    r"\bformerly\s+(?:known\s+as\s+)?(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
# Removed: "to stay up to date on X" (matched_pattern "stay_up_to_date") used
# to live here, scored 0.5. Pulled entirely after manual review found it was
# 100% false positive across every sample checked - 5/5 across two separate
# verification passes, with zero true positives observed. Unlike is_now
# (which had a real minority of genuine catches worth weighing against the
# noise), this pattern had nothing to trade off, so it's a straight removal
# rather than a confidence demotion.
#
# Removed: "X is now Y" / "X are now Y" (matched_pattern "is_now") used to
# live here, scored at the top confidence tier (1.0) alongside was_acquired_by.
# Pulled entirely after auditing all 2,316 is_now rows in
# combined_matching/acquired_companies.csv: ~87% were garbage-text captures
# ("is now available", "is now closed", "is now co-owned by Sonny and their
# partner" - sentence continuations, not company names), the worst false-
# positive rate of any pattern in the pipeline despite carrying its highest
# confidence score. A minority of genuine catches (e.g. "Network for Good is
# now Bonterra") are lost with it; some of those may still surface via the
# formerly fallback below at lower confidence, but not all will.

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
    ("formerly", _FORMERLY),
]


def _clean_acquirer(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip().strip(")").strip()


# --- Confidence scoring -------------------------------------------------------
#
# Base score per matched_pattern, 0.0-1.0. This is the only place these
# numbers live - adjust here, not per call site. "formerly" is included for
# completeness/documentation even though it's currently unreachable as a
# scored positive match (extract_acquirer_from_description() never returns
# an acquirerCompany for a standalone "formerly" - see that function's
# docstring); if that demotion is ever revisited, the score is already here.
#
# Cross-source modifiers (corroboration/conflict against the companyName-
# based path) are NOT applied here - this module only sees the description
# field in isolation, so that logic lives in combined_matching/combined_check.py,
# the one place both detectors' outputs are visible together.
_PATTERN_BASE_CONFIDENCE: dict[str, float] = {
    "was_acquired_by": 1.0,
    "wholly_owned_subsidiary": 1.0,
    "joined_forces": 0.7,
    "formerly": 0.3,
}


def _base_confidence_for_pattern(matched_pattern: str) -> float:
    return _PATTERN_BASE_CONFIDENCE.get(matched_pattern, 0.5)


def extract_acquirer_from_description(description: str | None) -> dict[str, Any] | None:
    """Scan `description` for acquisition-indicating phrases.

    Tries patterns in priority order: was_acquired_by > wholly_owned_subsidiary
    > joined_forces > formerly (supporting-only, see below). Returns None if
    there's no description or nothing
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
            "confidenceScore": _base_confidence_for_pattern("formerly"),
            "nameChanged": True,
            "isLikelyAcquirer": True,
            "rawText": formerly_match.group(0) if formerly_match else "formerly",
        }

    for pattern_name, pattern in (
        ("was_acquired_by", _WAS_ACQUIRED_BY),
        ("wholly_owned_subsidiary", _WHOLLY_OWNED_SUBSIDIARY),
        ("joined_forces", _JOINED_FORCES),
    ):
        m = pattern.search(text)
        if m and pattern_name == "joined_forces" and (
            _is_joined_forces_list_announcement(text, m)
            or _is_non_acquirer_organization(m.group("acquirer"))
            or _is_garbage_acquirer_text(m.group("acquirer"))
        ):
            # Partnership/client-list announcement, known non-acquirer org
            # (franchise network/accelerator/fund), or over-captured prose -
            # not a confident single acquirer either way. Discard entirely
            # (don't score, don't return) rather than picking one name out;
            # the remaining patterns below still get a chance to match.
            m = None
        if m:
            return {
                "acquirerCompany": _clean_acquirer(m.group("acquirer")),
                "matchedPattern": pattern_name,
                "confidence": "high",
                "confidenceScore": _base_confidence_for_pattern(pattern_name),
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
            "confidenceScore": _base_confidence_for_pattern("formerly"),
            "nameChanged": True,
            "isLikelyAcquirer": False,
            "rawText": formerly_match.group(0),
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
        "confidence_score": None,
        "confidence_notes": None,
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
        # No cross-source modifiers here - this module only sees the
        # description field, so confidence_notes stays None unless a future
        # caller (e.g. combined_check.py) recomputes it with more context.
        "confidence_score": match["confidenceScore"],
        "confidence_notes": None,
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
            "confidence_score": None,
            "confidence_notes": None,
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
    confidence_score: float | None
    confidence_notes: str | None
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
