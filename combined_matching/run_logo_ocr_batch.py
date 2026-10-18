"""Run ONLY the logo-OCR stage (stage 4) over EVERY company in sellers_0726
that has categoriesLLM2 populated - the exact same ~800k-document population
run_full_batch.py already ran stages 1-3 against (same `categoriesLLM2.keyword`
exists filter, same dwhId.keyword search_after pagination/checkpoint pattern).

Deliberately standalone, not merged into combined_acquisitions: writes into
its OWN table (logo_ocr_results) in the SAME db file stages 1-3 already wrote
to (DB_PATH below, matching run_full_batch.py's own DB_PATH). No corroborate/
conflict logic runs here, and combined_acquisitions is never touched by this
script - that merge is a deliberately separate, later step (a confidence-
combination rule against both tables, not yet written).

Reuses everything unmodified rather than reimplementing:
- logo_detection.py's _fetch_worker()/_ocr_worker() (the exact same fetch_logo_image/
  run_ocr_array/extract_acquirer_from_logo_text calls run_pipeline() itself
  drives) - but the thread pool built from them here is PERSISTENT for the
  whole run instead of being recreated per page - see the FIX note below.
- run_full_batch.py's ES paging shape (_EXISTS_QUERY on categoriesLLM2.keyword,
  sort by dwhId.keyword, search_after, retry-on-search) - same population,
  same resilience against the cluster's known intermittent 404 flakiness.

FIX (this module originally called logo_detection.run_pipeline() once PER
PAGE): run_pipeline() creates a brand new thread pool - including a brand
new OCR worker thread - on every call. ocr.py's PaddleOCR instance cache is
keyed by THREAD IDENTITY (threading.local()), so a brand new OCR thread
means a brand new ~528MB PaddleOCR instance loaded from scratch, every
single page. Confirmed via the log (a "Creating model" triple appeared once
per page, not once for the whole run) and via dmesg (the process was
OOM-killed by the kernel after ~10 pages on this box's ~1.8GB RAM). The fix
below builds the fetch/OCR thread pool ONCE in main() and feeds it every
page's work through the same long-lived queues - functionally identical
gating to run_pipeline() (nothing from page N+1 is enqueued until page N's
results have all been collected, so checkpointing after each page is still
exactly as safe as before), just without tearing down and rebuilding the
OCR thread - and therefore the OCR model - between pages.

Checkpointed the same way as run_full_batch.py: the last page's search_after
value + processed count are written to CHECKPOINT_PATH after every page, so a
crash or Ctrl+C resumes from where it left off rather than restarting - at an
estimated ~2.6 days of single-process OCR (see the conversation this was
built from), hitting an interruption at least once is likely, not an edge case.

Run with: python run_logo_ocr_batch.py
Resume an interrupted run: just run it again - CHECKPOINT_PATH picks up
where it left off automatically.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching", "logo_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_source import _normalize_text_field, normalize_company_source  # noqa: E402 (from name_matching)

from logo_detection import (  # noqa: E402
    FETCH_WORKERS,
    OCR_WORKERS,
    _OCR_QUEUE_MAXSIZE,
    _fetch_worker,
    _ocr_worker,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

# Same db file stages 1-3 already wrote to (run_full_batch.py's own DB_PATH) -
# a NEW table in it, combined_acquisitions itself is never touched here.
DB_PATH = "combined_acquisitions_full.db"
CHECKPOINT_PATH = "logo_ocr_batch_checkpoint.json"

PAGE_SIZE = 1000
PROGRESS_EVERY = 5000

_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2

# Outer-loop retry around fetch_page() in run_batch() - see the comment at
# its call site. Longer/more patient than _search_with_retry above: that one
# guards a single es.search() call, this one guards the whole page fetch
# against a run of blips that outlasts 3 quick retries.
_OUTER_FETCH_MAX_ATTEMPTS = 10
_OUTER_FETCH_RETRY_DELAY_SECONDS = 15

# Same population run_full_batch.py already ran stages 1-3 against.
_EXISTS_QUERY = {"exists": {"field": "categoriesLLM2.keyword"}}
_PAGE_SOURCE_FIELDS = ["dwhId", "companyName", "logoS3Path"]


def _search_with_retry(es: Elasticsearch, body: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            return es.search(index=ES_INDEX, body=body)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_SEARCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # type: ignore[misc]


def _load_checkpoint() -> dict[str, Any]:
    path = Path(CHECKPOINT_PATH)
    if not path.exists():
        return {"search_after": None, "processed": 0}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(search_after: list[Any] | None, processed: int) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"search_after": search_after, "processed": processed}, f)


def fetch_page(
    es: Elasticsearch, search_after: list[Any] | None
) -> tuple[list[dict[str, Any]], list[Any] | None, int]:
    """Returns (usable lookups, ES's own "sort" value for this page's last
    hit, raw hit count) - same shape/discipline as run_full_batch.py's own
    fetch_page(): pagination always advances using ES's own sort value, even
    past a hit with no usable dwhId, never a value re-extracted from
    _source."""
    body: dict[str, Any] = {
        "size": PAGE_SIZE,
        "query": _EXISTS_QUERY,
        "sort": [{"dwhId.keyword": "asc"}],
        "_source": _PAGE_SOURCE_FIELDS,
    }
    if search_after is not None:
        body["search_after"] = search_after

    resp = _search_with_retry(es, body)
    hits = resp["hits"]["hits"]

    sources = []
    for hit in hits:
        source = hit.get("_source")
        if not source or not source.get("dwhId"):
            logger.warning(
                "hit _id=%s has no usable _source/dwhId - skipping for OCR, "
                "still advancing pagination past it",
                hit.get("_id"),
            )
            continue
        # Same list-shaped-field surprise run_full_batch.py already hit and
        # fixed for companyName (normalize_company_source, from
        # name_matching/data_source.py, reused unmodified here) - crashed a
        # prior run of this script with sqlite3.ProgrammingError: type
        # 'list' is not supported (a real dwhId's companyName came back as
        # an ES array). logoS3Path isn't in normalize_company_source's own
        # field list (that module never reads it), so it gets the same
        # treatment via the same underlying helper directly.
        normalize_company_source(source)
        source["logoS3Path"] = _normalize_text_field(source.get("logoS3Path"))
        sources.append(source)

    last_sort = hits[-1]["sort"] if hits else None
    return sources, last_sort, len(hits)


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS logo_ocr_results (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            logoUrl TEXT,
            fetchOutcome TEXT,
            ocrText TEXT,
            matchedPattern TEXT,
            acquirerCompany TEXT,
            confidenceScore REAL,
            rawText TEXT
        )
        """
    )
    conn.commit()


def _to_sqlite_value(value: Any) -> Any:
    """SQLite only binds None/int/float/str/bytes. Coerce anything else to a
    string instead of letting conn.execute() raise sqlite3.ProgrammingError -
    same backstop run_full_batch.py already applies for this exact ES
    list-shaped-field surprise (see fetch_page()'s normalize_company_source
    call above, which is the primary fix; this is the belt-and-suspenders
    catch for any field that fix doesn't cover)."""
    if value is None or isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


def _write_page_results(conn: sqlite3.Connection, results: list[dict[str, Any]]) -> dict[str, int]:
    outcome_counts: dict[str, int] = {}
    for row in results:
        outcome_counts[row["fetch_outcome"]] = outcome_counts.get(row["fetch_outcome"], 0) + 1
        logo_result = row["logo_result"]
        ocr_text = row.get("ocr_text")
        conn.execute(
            "INSERT OR REPLACE INTO logo_ocr_results "
            "(dwhId, companyName, logoUrl, fetchOutcome, ocrText, matchedPattern, acquirerCompany, confidenceScore, rawText) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _to_sqlite_value(row["dwhId"]),
                _to_sqlite_value(row["companyName"]),
                _to_sqlite_value(row["logo_url"]),
                _to_sqlite_value(row["fetch_outcome"]),
                _to_sqlite_value(ocr_text),
                _to_sqlite_value(logo_result["matchedPattern"] if logo_result else None),
                _to_sqlite_value(logo_result["acquirerCompany"] if logo_result else None),
                _to_sqlite_value(logo_result["confidenceScore"] if logo_result else None),
                _to_sqlite_value(logo_result["rawText"] if logo_result else None),
            ),
        )
    conn.commit()
    return outcome_counts


def run_batch(es: Elasticsearch, conn: sqlite3.Connection) -> None:
    checkpoint = _load_checkpoint()
    search_after = checkpoint["search_after"]
    processed = checkpoint["processed"]
    fetch_outcome_totals: dict[str, int] = {}
    match_count = 0

    if search_after is not None:
        logger.info("Resuming from checkpoint: %d companies already processed", processed)

    # Persistent pool for the WHOLE run - built once, fed every page, never
    # torn down/rebuilt in between (see module docstring's FIX note for why
    # rebuilding per page OOM-killed the previous run). Gating is what keeps
    # per-page checkpointing exactly as safe as calling run_pipeline() per
    # page was: nothing from page N+1 is put on work_queue until page N's
    # len(page) results have all landed in `results`, so "wait for N more
    # results" always means exactly this page's results, never a mix.
    work_queue: "queue.Queue" = queue.Queue()
    ocr_queue: "queue.Queue" = queue.Queue(maxsize=_OCR_QUEUE_MAXSIZE)
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    progress = [0]  # cosmetic - only used by _ocr_worker's own "processed X/Y" print

    fetch_threads = [
        threading.Thread(target=_fetch_worker, args=(work_queue, ocr_queue), daemon=True)
        for _ in range(FETCH_WORKERS)
    ]
    ocr_threads = [
        threading.Thread(target=_ocr_worker, args=(ocr_queue, results, results_lock, progress, 0), daemon=True)
        for _ in range(OCR_WORKERS)
    ]
    for t in fetch_threads + ocr_threads:
        t.start()

    start = time.time()
    try:
        while True:
            # Outer retry, on top of fetch_page -> _search_with_retry's own
            # 3-attempt/2s inner retry: this cluster is documented (see
            # es_client.py's docstring) to intermittently return a transient
            # index_not_found_exception even though the index exists, and a
            # prior run of this script died to exactly that once the inner
            # retry's 3 attempts all landed inside one such blip. At
            # multi-day scale that's not a one-off - the outer loop here
            # keeps retrying the SAME search_after (nothing's checkpointed
            # yet, so no risk of skipping/duplicating a page) instead of
            # letting one bad blip kill the whole run.
            for outer_attempt in range(1, _OUTER_FETCH_MAX_ATTEMPTS + 1):
                try:
                    page, last_sort, raw_hit_count = fetch_page(es, search_after)
                    break
                except Exception as exc:
                    logger.warning(
                        "fetch_page failed (outer attempt %d/%d): %s",
                        outer_attempt, _OUTER_FETCH_MAX_ATTEMPTS, exc,
                    )
                    if outer_attempt == _OUTER_FETCH_MAX_ATTEMPTS:
                        raise
                    time.sleep(_OUTER_FETCH_RETRY_DELAY_SECONDS)

            if raw_hit_count == 0:
                break

            start_len = len(results)
            for source in page:
                work_queue.put(
                    (
                        source["dwhId"],
                        source.get("companyName"),
                        (source.get("logoS3Path") or "").strip() or None,
                    )
                )

            # Block until exactly this page's results have all landed -
            # cheap poll, fine at ~4-5 minutes/page.
            target_len = start_len + len(page)
            while True:
                with results_lock:
                    current_len = len(results)
                if current_len >= target_len:
                    break
                time.sleep(0.5)
            with results_lock:
                page_results = results[start_len:target_len]

            page_outcomes = _write_page_results(conn, page_results)
            for outcome, count in page_outcomes.items():
                fetch_outcome_totals[outcome] = fetch_outcome_totals.get(outcome, 0) + count
            match_count += sum(1 for r in page_results if r["logo_result"] is not None)

            processed += len(page)
            search_after = last_sort
            _save_checkpoint(search_after, processed)

            if processed % PROGRESS_EVERY < PAGE_SIZE:
                elapsed = time.time() - start
                logger.info(
                    "Processed %d companies so far (%.1fs elapsed, %d matches, outcomes=%s)",
                    processed, elapsed, match_count, fetch_outcome_totals,
                )

            if raw_hit_count < PAGE_SIZE:
                break
    finally:
        # Same shutdown sequence as logo_detection.run_pipeline(), just done
        # once here instead of once per page.
        for _ in range(FETCH_WORKERS):
            work_queue.put(None)
        for t in fetch_threads:
            t.join()
        for _ in range(OCR_WORKERS):
            ocr_queue.put(None)
        for t in ocr_threads:
            t.join()

    elapsed = time.time() - start
    logger.info("Done. Processed %d companies total in %.1fs.", processed, elapsed)
    logger.info("Fetch outcomes: %s", fetch_outcome_totals)
    logger.info("Logo matches (acquirer text found): %d", match_count)


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)
    conn = sqlite3.connect(DB_PATH)
    _create_table(conn)

    try:
        run_batch(es, conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
