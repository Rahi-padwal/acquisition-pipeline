"""Audit a batch of already-flagged-acquired companies (isAcquired=1 in
combined_with_ocr.db) by independently re-checking all 4 detection stages -
company name (name_matching), description (description_matching), S3 web
scrape (s3_detection), logo OCR (logo_matching, reused from the DB rather
than re-run) - and scoring each company by how many of those 4 sources
actually corroborate the same acquirer, per this spec:

  - ES's own isAcquired flag + a usable structured acquiredBy field
    (check_name_signals._check_structured_fields) -> confidenceScore = 1.0
    outright, no further checks needed.
  - Otherwise, take whichever acquirer name any ONE stage's own pattern
    found, then check the OTHER 3 sources' raw text for that SAME name
    (fuzzy match) instead of re-running their own patterns. Count how many
    of the 4 sources' raw text contains it:
        >= 3 sources -> confidenceScore = 1.0
        == 2 sources -> confidenceScore = 0.5
        == 1 source  -> confidenceScore = 0.3
        0 sources    -> confidenceScore = 0.0 (no acquirer found at all)
  - A row is flagged FP (false positive of the existing isAcquired=1 label)
    when confidenceScore <= 0.3.

Input: an .xlsx with a "dwhId" column (e.g. "Untitled spreadsheet (1).xlsx").
Output: an .xlsx with one row per company - companyName, acquirerCompany,
confidenceScore, FP, plus dwhId and matchedStages for traceability.

Usage:
    python confidence_scoring.py "Untitled spreadsheet (1).xlsx" [output.xlsx]
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from rapidfuzz import fuzz

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import check_company_record  # noqa: E402
from data_source import ElasticCompanyDataSource  # noqa: E402

from s3_detection import (  # noqa: E402
    S3_BUCKET_NAME,
    S3_PREFIX,
    _s3_client,
    combined_text_blob_multi,
    fetch_all_scrapes,
    list_pipeline_folders,
)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

ES_WORKERS = 10
S3_WORKERS = 64

# User is checking S3 corroboration manually for now - skip the live S3
# fetch stage entirely rather than waiting on it. Rows whose original
# acquirer came from s3_scrape lose their one self-matching source when
# this is on, so they'll show matchedStages="" / FP=True unless name,
# description, or logo also happen to corroborate them - expected, not a
# bug, given S3 is being checked out of band.
SKIP_S3 = True

SOURCE_DB_PATH = str(Path(__file__).resolve().parent / "combined_with_ocr.db")

DEFAULT_OUTPUT_XLSX = "confidence_scoring_output.xlsx"

# Order the matchedStages column lists corroborating sources in.
_STAGE_PRIORITY = ["name", "description", "s3", "logo"]

_FUZZY_THRESHOLD = 85  # tolerate a 1-2 char OCR/spelling slip (e.g. "Cimarron" vs "Cimarran" = 87.5)
_MIN_FUZZY_LEN = 4  # shorter normalized names: exact substring match only
_MAX_FUZZY_WINDOW_WORDS = 15_000  # cap the sliding-window FUZZY fallback's cost on huge S3 blobs

_SUFFIX_WORDS = {
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "group", "gmbh", "plc", "sa", "srl", "bv", "ag",
    "holdings", "holding", "the",
}


def _normalize(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = [w for w in text.split() if w not in _SUFFIX_WORDS]
    return " ".join(words)


def _name_found_in_text(acquirer: str | None, text: str | None) -> bool:
    """Fuzzy/substring check: does `acquirer` appear in `text`? Used both
    for a stage's OWN raw text (trivially true - that's where it was
    extracted from) and for cross-checking the OTHER 3 stages' raw text."""
    if not acquirer or not text:
        return False
    needle = _normalize(acquirer)
    if not needle:
        return False
    haystack = _normalize(text)
    if not haystack:
        return False
    if needle in haystack:  # exact substring check runs against the FULL text, no truncation
        return True
    if len(needle) < _MIN_FUZZY_LEN:
        return False  # avoid noisy fuzzy hits on very short names

    needle_words = needle.split()
    haystack_words = haystack.split()[:_MAX_FUZZY_WINDOW_WORDS]  # only the expensive fuzzy fallback is capped
    n = len(needle_words)
    if len(haystack_words) < n:
        return False

    best = 0
    for i in range(len(haystack_words) - n + 1):
        window = " ".join(haystack_words[i : i + n])
        score = fuzz.ratio(needle, window)
        if score > best:
            best = score
            if best >= 100:
                break
    return best >= _FUZZY_THRESHOLD


# A single-source s3-only 0.3 hit is otherwise treated as trustworthy (the
# company's own scraped site directly stating the acquisition) - EXCEPT
# when the original match came via the 'provided_by' pattern, which the
# codebase's own docs flag as unguarded against GDPR/cookie-policy
# boilerplate ("...webanalysedienst van Google Inc"), or when the acquirer
# name itself is a well-known analytics/embed/tool vendor rather than a
# real acquirer - both are confirmed false-positive shapes, not real
# corroboration gaps.
_RISKY_ORIGINAL_PATTERNS = {"provided_by"}
_KNOWN_NON_ACQUIRERS = (
    "google", "vimeo", "youtube", "facebook", "meta platforms", "linkedin",
    "twitter", "cloudflare", "amazon web services", "mailchimp",
    "rocket science group", "hubspot", "wordpress", "automattic", "shopify",
    "wix", "squarespace", "godaddy", "microsoft", "stripe", "paypal",
    "hotjar", "mixpanel", "segment", "sendgrid", "twilio",
)


def _is_known_non_acquirer(acquirer: str | None) -> bool:
    normalized = _normalize(acquirer)
    return any(token in normalized for token in _KNOWN_NON_ACQUIRERS)


def _is_false_positive(
    original_acquired: bool,
    original_matched_pattern: str | None,
    acquirer: str | None,
    confidence_score: float,
    matched_stages: list[str],
) -> bool:
    if not original_acquired or confidence_score > 0.3:
        return False
    if matched_stages == ["s3"]:
        if original_matched_pattern in _RISKY_ORIGINAL_PATTERNS or _is_known_non_acquirer(acquirer):
            return True
        return False  # single-source s3 is otherwise trusted, not flagged FP
    return True


def _read_dwh_ids(path: str) -> list[str]:
    df = pd.read_excel(path, dtype=str)
    col = "dwhId" if "dwhId" in df.columns else df.columns[0]
    return [v.strip() for v in df[col].dropna().tolist() if v.strip()]


def _load_existing(dwh_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Pull the existing isAcquired label plus already-computed logo OCR
    result (acquirerCompany/matchedPattern/ocrText) for each dwhId - stage 4
    was already run for this whole set, so it's reused rather than re-run."""
    conn = sqlite3.connect(SOURCE_DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(dwh_ids))

    existing: dict[str, dict[str, Any]] = {}
    q = (
        "SELECT dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern "
        f"FROM combined_acquisitions WHERE dwhId IN ({placeholders})"
    )
    for row in conn.execute(q, dwh_ids).fetchall():
        existing[row["dwhId"]] = {
            "originalIsAcquired": row["isAcquired"],
            "originalCompanyName": row["companyName"],
            "originalAcquirerCompany": row["acquirerCompany"],
            "originalSource": row["source"],
            "originalMatchedPattern": row["matchedPattern"],
        }

    q2 = f"SELECT dwhId, acquirerCompany, ocrText FROM logo_ocr_results WHERE dwhId IN ({placeholders})"
    for row in conn.execute(q2, dwh_ids).fetchall():
        existing.setdefault(row["dwhId"], {})
        existing[row["dwhId"]]["logoAcquirer"] = row["acquirerCompany"]
        existing[row["dwhId"]]["logoText"] = row["ocrText"]

    conn.close()
    return existing


def _run_es_stage(dwh_id: str, data_source: ElasticCompanyDataSource) -> dict[str, Any]:
    try:
        record = data_source.get_by_dwh_id(dwh_id)
    except Exception as exc:
        print(f"  dwhId {dwh_id}: ES fetch failed - {exc}")
        record = None

    if record is None:
        return {
            "dwhId": dwh_id, "companyName": None, "companyNameText": "",
            "descriptionText": "", "structuredAcquirer": None,
        }

    name_match = check_company_record(record)
    structured_acquirer = (
        name_match["candidate_name"]
        if name_match is not None and name_match["matched_pattern"] == "structured_field"
        else None
    )

    return {
        "dwhId": dwh_id,
        "companyName": record.company_name,
        "companyNameText": record.company_name or "",
        "descriptionText": record.raw.get("description") or "",
        "structuredAcquirer": structured_acquirer,
    }


def _run_s3_stage(s3, folders: list[str], dwh_id: str) -> dict[str, Any]:
    scrapes, _ = fetch_all_scrapes(s3, S3_BUCKET_NAME, folders, dwh_id)
    blob = combined_text_blob_multi(scrapes) if scrapes else ""
    return {"dwhId": dwh_id, "s3Text": blob}


def _score_company(
    es_row: dict[str, Any], s3_row: dict[str, Any], existing_row: dict[str, Any]
) -> dict[str, Any]:
    """Per the spec: don't re-derive a fresh acquirer candidate per stage -
    take the acquirer name ALREADY on record (from whichever stage the
    original pipeline found it through) and check the other sources' raw
    text for that SAME name, instead of each stage's own pattern."""
    dwh_id = es_row["dwhId"]
    company_name = es_row["companyName"] or existing_row.get("originalCompanyName")
    original_acquired = bool(existing_row.get("originalIsAcquired"))

    if es_row["structuredAcquirer"]:
        return {
            "dwhId": dwh_id,
            "companyName": company_name,
            "acquirerCompany": es_row["structuredAcquirer"],
            "confidenceScore": 1.0,
            "matchedStages": "elastic_structured",
            "FP": False,
        }

    acquirer = existing_row.get("originalAcquirerCompany")
    if not acquirer:
        return {
            "dwhId": dwh_id,
            "companyName": company_name,
            "acquirerCompany": None,
            "confidenceScore": 0.0,
            "matchedStages": "",
            "FP": original_acquired,
        }

    texts = {
        "name": es_row["companyNameText"],
        "description": es_row["descriptionText"],
        "s3": s3_row["s3Text"],
        "logo": existing_row.get("logoText") or "",
    }
    matched = [stage for stage in _STAGE_PRIORITY if _name_found_in_text(acquirer, texts[stage])]
    count = len(matched)

    if count >= 3:
        score = 1.0
    elif count == 2:
        score = 0.5
    elif count == 1:
        score = 0.3
    else:
        score = 0.0

    fp = _is_false_positive(
        original_acquired, existing_row.get("originalMatchedPattern"), acquirer, score, matched
    )
    return {
        "dwhId": dwh_id,
        "companyName": company_name,
        "acquirerCompany": acquirer,
        "confidenceScore": score,
        "matchedStages": "+".join(matched),
        "FP": fp,
    }


def main(input_path: str, output_xlsx: str) -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    existing = _load_existing(dwh_ids)
    print(f"Loaded existing isAcquired/logo data for {len(existing)} dwhIds")

    print("\nStage 1+2 (companyName + description, live ES)...")
    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)
    es_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=ES_WORKERS) as executor:
        futures = {executor.submit(_run_es_stage, dwh_id, data_source): dwh_id for dwh_id in dwh_ids}
        done = 0
        for future in as_completed(futures):
            row = future.result()
            es_results[row["dwhId"]] = row
            done += 1
            if done % 500 == 0:
                print(f"  ES stage: {done}/{len(dwh_ids)}")
    print(f"ES stage done: {len(es_results)}/{len(dwh_ids)}")

    s3_results: dict[str, dict[str, Any]] = {}
    if SKIP_S3:
        print("\nStage 3 (S3 web scrape) - SKIPPED (SKIP_S3=True, checking manually out of band)")
        s3_results = {dwh_id: {"dwhId": dwh_id, "s3Text": ""} for dwh_id in dwh_ids}
    else:
        print("\nStage 3 (S3 web scrape, live)...")
        s3 = _s3_client()
        folders = list_pipeline_folders(s3, S3_BUCKET_NAME, S3_PREFIX)
        with ThreadPoolExecutor(max_workers=S3_WORKERS) as executor:
            futures = {
                executor.submit(_run_s3_stage, s3, folders, dwh_id): dwh_id
                for dwh_id in dwh_ids
            }
            done = 0
            for future in as_completed(futures):
                row = future.result()
                s3_results[row["dwhId"]] = row
                done += 1
                if done % 500 == 0:
                    print(f"  S3 stage: {done}/{len(dwh_ids)}")
        print(f"S3 stage done: {len(s3_results)}/{len(dwh_ids)}")

    print("\nScoring...")
    output_rows = []
    for dwh_id in dwh_ids:
        es_row = es_results.get(dwh_id) or {
            "dwhId": dwh_id, "companyName": None, "companyNameText": "",
            "descriptionText": "", "structuredAcquirer": None,
        }
        s3_row = s3_results.get(dwh_id) or {"dwhId": dwh_id, "s3Text": ""}
        existing_row = existing.get(dwh_id, {})
        output_rows.append(_score_company(es_row, s3_row, existing_row))

    df_out = pd.DataFrame(
        output_rows,
        columns=["dwhId", "companyName", "acquirerCompany", "confidenceScore", "matchedStages", "FP"],
    )
    df_out.to_excel(output_xlsx, index=False)

    fp_count = int(df_out["FP"].sum())
    print(f"\nWrote {len(df_out)} rows to {output_xlsx}")
    print(f"  false positives (FP=True): {fp_count} of {len(df_out)}")
    print(df_out["confidenceScore"].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python confidence_scoring.py "input.xlsx" [output.xlsx]')
        sys.exit(1)
    _input_path = sys.argv[1]
    _output_xlsx = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_XLSX
    main(_input_path, _output_xlsx)
