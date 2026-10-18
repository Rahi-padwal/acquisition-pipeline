"""Stage-4 acquisition detection: text extracted via OCR from a company's
own LinkedIn-scraped logo image.

A FOURTH, independent detection source alongside company-name
(name_matching/check_name_signals.py), description
(description_matching/description_detection_api.py), and S3 web-scrape
(s3_detection.py). Those three only ever see TEXT fields already sitting in
Elasticsearch or S3; this one reads the logo GRAPHIC itself - companies
occasionally render their acquirer's name directly into the logo artwork
("Company X, a Y Company", "Powered by Y"), which none of the other three
sources can ever see since it never appears as plain text anywhere in ES.

Reuses logo_matching/'s already-built-and-tested modules unmodified (own
pattern set in detection.py, own PaddleOCR wrapper in ocr.py, own ES lookup
in es_client.py, own case-handled HTTP fetch in fetch_logo.py) rather than
reimplementing them here - see logo_matching/detection.py's docstring for
why its 5-pattern set (a_x_company, part_of, x_group, powered_by, by_x)
differs from both this module's own siblings' pattern sets.

Validated before being wired in here: check_against_known.py (in
logo_matching/) ran this exact detection logic against the 20 dwhIds
stages 1-3 had ALREADY confirmed an acquirer for for. Result: 3/20 logos
had any acquirer text at all, and all 3 exactly matched the already-known
acquirer (0 conflicts). Small sample, but zero false positives against
known answers is why this was judged safe to add rather than left as a
standalone verification script.

Deliberately sequential, not threaded like s3_detection.py's
MAX_WORKERS=64: logo_matching/ocr.py's PaddleOCR instance is a
module-level singleton (see its _get_ocr()) never tested for thread
safety, and OCR inference itself is CPU-bound (unlike s3_detection.py's
pure-I/O S3 GETs), so naive threading here isn't a safe drop-in of that
same pattern. Revisit if/when this needs to run at real batch scale -
options include a lock around the shared OCR call, or a small
process/thread pool sized to actual CPU headroom, but that's an explicit
follow-up decision, not made here.

Run with: python logo_detection.py [input.csv] [db_path]
Input file: one dwhId per row, no header (same convention as
run_pipeline_on_file.py / s3_detection.py). Defaults to "test data.csv" /
"test_combined_pipeline.db", matching s3_detection.py's own defaults.

Deliberately NOT wired into run_full_batch.py's live 768k-company loop as
part of this change - that's a separate, larger decision (this script
mirrors exactly the standalone stage s3_detection.py started as, before it
was later wired into run_full_batch.py as stage 3).
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from pathlib import Path
from sqlite3 import Connection, Row, connect
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching", "logo_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import _PATTERN_BASE_CONFIDENCE as _NAME_PATTERN_BASE_CONFIDENCE  # noqa: E402
from description_detection_api import _PATTERN_BASE_CONFIDENCE as _DESCRIPTION_PATTERN_BASE_CONFIDENCE  # noqa: E402
from detection import _PATTERN_BASE_CONFIDENCE as _LOGO_PATTERN_BASE_CONFIDENCE  # noqa: E402 (from logo_matching)
from detection import extract_acquirer_from_logo_text  # noqa: E402 (from logo_matching)
from es_client import fetch_logo_lookups  # noqa: E402 (from logo_matching)
from fetch_logo import fetch_logo_image  # noqa: E402 (from logo_matching)
from ocr import run_ocr_array  # noqa: E402 (from logo_matching)

from combined_check import (  # noqa: E402
    _CONFLICT_CAP,
    _CORROBORATION_BONUS,
    _CORROBORATION_CAP,
    _HIGH_TRUST_CONFLICT_CAP,
    _HIGH_TRUST_MIN_BASE_SCORE,
    _company_identity_match,
    classify_review_status,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SOURCE_NAME = "logo_ocr"

DEFAULT_INPUT_PATH = "test data.csv"
DEFAULT_DB_PATH = "test_combined_pipeline.db"
DEFAULT_OUTPUT_XLSX = "logo_detection_output.xlsx"

OUTPUT_COLUMNS = [
    "dwhId", "companyName", "isAcquired", "acquirerCompany", "source",
    "matchedPattern", "confidenceScore", "reviewStatus",
    "logoFetchOutcome", "logoMatchedPattern", "logoAcquirer", "logoRawText", "logoUrl",
]


# --- Cross-source corroboration/conflict, applied against an EXISTING row ----
#
# Same duplication rationale as s3_detection.py's own copy of this logic:
# combined_check.py's _is_high_trust_pattern only checks name_matching's and
# description_matching's pattern tables, and logo_matching's own pattern
# names (a_x_company, part_of, x_group, powered_by, by_x) don't exist in
# either - a_x_company in particular is scored 0.9 (high-trust) here but
# would silently fall through to the low-trust conflict cap without this.
def _is_high_trust_pattern(matched_pattern: str) -> bool:
    base = (
        _NAME_PATTERN_BASE_CONFIDENCE.get(matched_pattern)
        or _DESCRIPTION_PATTERN_BASE_CONFIDENCE.get(matched_pattern)
        or _LOGO_PATTERN_BASE_CONFIDENCE.get(matched_pattern)
    )
    return base is not None and base >= _HIGH_TRUST_MIN_BASE_SCORE


def _conflict_capped_score(score: float, matched_pattern: str) -> float:
    cap = _HIGH_TRUST_CONFLICT_CAP if _is_high_trust_pattern(matched_pattern) else _CONFLICT_CAP
    return min(score, cap)


def _apply_cross_source_modifiers(
    primary_acquirer: str, primary_score: float, other_acquirer: str, primary_pattern: str
) -> tuple[float, str]:
    if _company_identity_match(primary_acquirer, other_acquirer):
        return min(_CORROBORATION_CAP, primary_score + _CORROBORATION_BONUS), "corroborated_by_both_sources"
    return _conflict_capped_score(primary_score, primary_pattern), "source_conflict"


# --- DB merge -------------------------------------------------------------


def _create_table_if_missing(conn: Connection) -> None:
    """Same schema every other stage in this project uses - this module
    never creates a new table, only adds rows/updates existing ones."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS combined_acquisitions (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            isAcquired INTEGER,
            acquirerCompany TEXT,
            source TEXT,
            matchedPattern TEXT,
            confidenceScore REAL,
            reviewStatus TEXT
        )
        """
    )
    conn.commit()


def merge_logo_result_into_db(
    conn: Connection, dwh_id: str, company_name: str | None, logo_result: dict[str, Any] | None
) -> str:
    """Write `logo_result` (or lack of one) for `dwh_id` into
    combined_acquisitions, honoring whatever row is already there. Returns
    a short outcome label for the run summary: "no_match" | "fresh_insert" |
    "upgraded_from_empty" | "corroborated" | "conflict_capped". Identical
    shape to s3_detection.py's merge_s3_result_into_db() - never silently
    overwrites an existing acquirer."""
    conn.row_factory = Row
    existing = conn.execute("SELECT * FROM combined_acquisitions WHERE dwhId = ?", (dwh_id,)).fetchone()

    if logo_result is None:
        if existing is None:
            is_acquired, review_status = classify_review_status(None)
            conn.execute(
                "INSERT INTO combined_acquisitions "
                "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
                "VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?)",
                (dwh_id, company_name, is_acquired, review_status),
            )
        return "no_match"

    acquirer, pattern, score = logo_result["acquirerCompany"], logo_result["matchedPattern"], logo_result["confidenceScore"]

    if existing is None or not existing["acquirerCompany"]:
        is_acquired, review_status = classify_review_status(score)
        conn.execute(
            "INSERT OR REPLACE INTO combined_acquisitions "
            "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (dwh_id, company_name, is_acquired, acquirer, SOURCE_NAME, pattern, score, review_status),
        )
        return "fresh_insert" if existing is None else "upgraded_from_empty"

    adjusted_score, notes = _apply_cross_source_modifiers(
        existing["acquirerCompany"], existing["confidenceScore"] or 0.0, acquirer, existing["matchedPattern"]
    )
    is_acquired, review_status = classify_review_status(adjusted_score)
    conn.execute(
        "UPDATE combined_acquisitions SET isAcquired = ?, confidenceScore = ?, reviewStatus = ? WHERE dwhId = ?",
        (is_acquired, adjusted_score, review_status, dwh_id),
    )

    if notes == "source_conflict":
        logger.warning(
            "dwhId %s: logo conflict - existing acquirer %r (%s) vs logo acquirer %r (%s) - "
            "keeping existing, score capped to %s",
            dwh_id, existing["acquirerCompany"], existing["matchedPattern"], acquirer, pattern, adjusted_score,
        )
        return "conflict_capped"

    logger.info(
        "dwhId %s: logo corroborates existing acquirer %r - score %s -> %s",
        dwh_id, existing["acquirerCompany"], existing["confidenceScore"], adjusted_score,
    )
    return "corroborated"


# --- Driver --------------------------------------------------------------


def _read_dwh_ids(path: str) -> list[str]:
    read = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = read(path, header=None, dtype=str)
    return [v.strip() for v in df.iloc[:, 0].dropna().tolist() if v.strip()]


# --- Concurrent fetch -> OCR pipeline -----------------------------------
#
# Two separate worker pools, not one shared pool doing fetch+OCR per task -
# the two stages have very different resource profiles and mixing them
# into one pool means the process would be sized for whichever is smaller,
# wasting the other:
#   - fetch is pure network I/O (HTTP GET from S3), cheap per worker, safe
#     at high concurrency - same reasoning as s3_detection.py's
#     MAX_WORKERS=64 for its own pure-S3-I/O stage, though kept more
#     conservative here since this runs on a shared dev machine, not a
#     dedicated crawler box.
#   - OCR is CPU-bound AND memory-hungry: each worker needs its OWN
#     PaddleOCR instance (see ocr.py's thread-local fix - a single shared
#     instance was never verified safe for concurrent calls), and each
#     instance measured ~414MB RSS on this project's models. OCR_WORKERS
#     is capped by available RAM, not CPU core count - 3 workers (~1.24GB)
#     was chosen to fit comfortably within the ~1.85GB free observed on
#     this machine during testing, not to saturate all 8 logical cores.
#
# ocr_queue is bounded (unlike work_queue, whose items are just tiny
# id/url strings) specifically so fetch workers can't race arbitrarily far
# ahead of the slower OCR stage and buffer unbounded decoded images in
# memory - at 850K scale that's a real risk a small test run would never
# surface. Fetch workers simply block on ocr_queue.put() once it's full,
# which is the backpressure this needs.
FETCH_WORKERS = 16
OCR_WORKERS = 1
_OCR_QUEUE_MAXSIZE = 20
_PROGRESS_EVERY = 50


def _fetch_worker(work_queue: "queue.Queue", ocr_queue: "queue.Queue") -> None:
    while True:
        item = work_queue.get()
        if item is None:
            return
        dwh_id, company_name, logo_url = item
        fetch_result = fetch_logo_image(logo_url)
        ocr_queue.put((dwh_id, company_name, logo_url, fetch_result))


def _ocr_worker(
    ocr_queue: "queue.Queue",
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    progress: list[int],
    total: int,
) -> None:
    while True:
        item = ocr_queue.get()
        if item is None:
            return
        dwh_id, company_name, logo_url, fetch_result = item

        if fetch_result.outcome != "ok":
            row = {
                "dwhId": dwh_id, "companyName": company_name, "logo_url": logo_url,
                "fetch_outcome": fetch_result.outcome, "logo_result": None,
            }
        else:
            ocr_result = run_ocr_array(fetch_result.image)
            logo_result = extract_acquirer_from_logo_text(ocr_result.high_confidence_text(), company_name)
            row = {
                "dwhId": dwh_id, "companyName": company_name, "logo_url": logo_url,
                "fetch_outcome": "ok", "logo_result": logo_result,
            }

        with results_lock:
            results.append(row)
            progress[0] += 1
            if progress[0] % _PROGRESS_EVERY == 0:
                print(f"  processed {progress[0]}/{total}")


def run_pipeline(dwh_ids: list[str], lookups: dict[str, Any]) -> list[dict[str, Any]]:
    """Fetch + OCR + extraction for every dwhId, concurrently - no DB
    access here (DB writes happen single-threaded afterward in main(),
    same discipline as s3_detection.py). Results come back in COMPLETION
    order, not input order - callers that need input order (the final
    Excel output in main()) already re-sort by dwhId, so this is safe."""
    work_queue: "queue.Queue" = queue.Queue()
    ocr_queue: "queue.Queue" = queue.Queue(maxsize=_OCR_QUEUE_MAXSIZE)

    for dwh_id in dwh_ids:
        lookup = lookups[dwh_id]
        work_queue.put((dwh_id, lookup.company_name, lookup.logo_s3_path))

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    progress = [0]

    fetch_threads = [
        threading.Thread(target=_fetch_worker, args=(work_queue, ocr_queue), daemon=True)
        for _ in range(FETCH_WORKERS)
    ]
    ocr_threads = [
        threading.Thread(target=_ocr_worker, args=(ocr_queue, results, results_lock, progress, len(dwh_ids)), daemon=True)
        for _ in range(OCR_WORKERS)
    ]
    for t in fetch_threads + ocr_threads:
        t.start()

    # One sentinel per fetch worker so each exits its loop exactly once,
    # after (FIFO) every real work item ahead of it has been consumed.
    for _ in range(FETCH_WORKERS):
        work_queue.put(None)
    for t in fetch_threads:
        t.join()

    # Every fetch thread has returned, which only happens after its final
    # ocr_queue.put() call - safe to signal OCR workers to stop now that
    # nothing more will ever be added to ocr_queue.
    for _ in range(OCR_WORKERS):
        ocr_queue.put(None)
    for t in ocr_threads:
        t.join()

    return results


def main(input_path: str, db_path: str, output_xlsx: str) -> None:
    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    print("Looking up logoS3Path in Elasticsearch...")
    lookups = fetch_logo_lookups(dwh_ids)

    print(f"Fetching + OCR'ing with {FETCH_WORKERS} fetch workers / {OCR_WORKERS} OCR workers...")
    results = run_pipeline(dwh_ids, lookups)

    conn = connect(db_path)
    _create_table_if_missing(conn)

    outcomes: dict[str, int] = {}
    fetch_outcome_counts: dict[str, int] = {}
    matched_samples: list[dict[str, Any]] = []
    for row in results:
        fetch_outcome_counts[row["fetch_outcome"]] = fetch_outcome_counts.get(row["fetch_outcome"], 0) + 1
        if row["fetch_outcome"] != "ok":
            row["outcome"] = row["fetch_outcome"]
            continue
        outcome = merge_logo_result_into_db(conn, row["dwhId"], row["companyName"], row["logo_result"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        row["outcome"] = outcome
        if row["logo_result"] is not None:
            matched_samples.append(row)

    conn.row_factory = Row
    placeholders = ",".join("?" * len(dwh_ids))
    db_rows = {
        r["dwhId"]: dict(r)
        for r in conn.execute(
            f"SELECT * FROM combined_acquisitions WHERE dwhId IN ({placeholders})", dwh_ids
        ).fetchall()
    }
    conn.commit()
    conn.close()

    print(f"\nProcessed {len(dwh_ids)} dwhIds")
    print("Fetch outcomes:")
    for label in ("ok", "no_logo", "access_denied", "fetch_failed"):
        print(f"  {label:<15} = {fetch_outcome_counts.get(label, 0)}")
    print("DB merge outcomes (of successfully fetched+OCR'd logos):")
    for label in ("no_match", "fresh_insert", "upgraded_from_empty", "corroborated", "conflict_capped"):
        print(f"  {label:<24} = {outcomes.get(label, 0)}")

    print(f"\nSample of matched rows for manual verification (up to 20 of {len(matched_samples)}):")
    for row in matched_samples[:20]:
        r = row["logo_result"]
        print(
            f"  dwhId={row['dwhId']} companyName={row['companyName']!r} outcome={row['outcome']} "
            f"pattern={r['matchedPattern']} score={r['confidenceScore']} acquirer={r['acquirerCompany']!r} "
            f"rawText={r['rawText']!r}"
        )

    output_rows = []
    for row in results:
        db_row = db_rows.get(row["dwhId"], {})
        logo_result = row["logo_result"]
        output_rows.append(
            {
                "dwhId": row["dwhId"],
                "companyName": db_row.get("companyName", row["companyName"]),
                "isAcquired": db_row.get("isAcquired"),
                "acquirerCompany": db_row.get("acquirerCompany"),
                "source": db_row.get("source"),
                "matchedPattern": db_row.get("matchedPattern"),
                "confidenceScore": db_row.get("confidenceScore"),
                "reviewStatus": db_row.get("reviewStatus"),
                "logoFetchOutcome": row["outcome"],
                "logoMatchedPattern": logo_result["matchedPattern"] if logo_result else None,
                "logoAcquirer": logo_result["acquirerCompany"] if logo_result else None,
                "logoRawText": logo_result["rawText"] if logo_result else None,
                "logoUrl": row["logo_url"],
            }
        )

    order = {dwh_id: i for i, dwh_id in enumerate(dwh_ids)}
    output_rows.sort(key=lambda r: order.get(r["dwhId"], len(dwh_ids)))
    df_out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    df_out.to_excel(output_xlsx, index=False)
    print(f"\nWrote {len(df_out)} rows to {output_xlsx}")


if __name__ == "__main__":
    _input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    _db_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB_PATH
    _output_xlsx = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_XLSX
    main(_input_path, _db_path, _output_xlsx)
