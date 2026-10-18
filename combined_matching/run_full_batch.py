"""Run the full 3-stage detection pipeline (companyName -> description ->
S3 web scrape) over EVERY company in sellers_0726 that has categoriesLLM2
populated - not a sample, the whole ~768k-document set (confirmed via a
_count query - see the conversation this was built from).

Stage 3 runs per-page, right after that page's stage 1+2 rows are
committed, reusing s3_detection.py's fetch_all_scrapes()/
extract_acquirer_from_s3_text()/merge_s3_result_into_db() unmodified - same
corroboration/conflict logic as everywhere else in this project. At this
scale (~768k companies x up to 6 S3 lookups each) stage 3 is the dominant
cost of a full run - budget for it accordingly (see the conversation this
was added from for a rough time estimate).

Unlike test_combined_pipeline.py's random_score sampling, this paginates
through the FULL matching set via search_after, sorted by dwhId.keyword
(dwhId itself is mapped `text`, not sortable directly - ES rejects sorting
on analyzed text fields; dwhId.keyword is the companion keyword sub-field,
same pattern as companyName.keyword / companyWebsiteDomain.keyword
elsewhere in this project). search_after always uses ES's own "sort" value
from the last hit in a page (see fetch_page()) - never a field re-read out
of that hit's own _source, which can be missing or shaped unexpectedly (a
raw document occasionally has no usable dwhId at all; such hits are logged
and skipped for detection, but pagination still advances past them using
the sort value ES already gave us for that hit). Checkpointed: that sort
value is written to CHECKPOINT_PATH after every page, so a crash or Ctrl+C
can be resumed by just running this again rather than restarting from page
1 - relevant at this scale, since ~768 page-fetches means the cluster's
known intermittent 404 flakiness (see _search_with_retry() elsewhere in
this project) is very likely to be hit at least once over a full run.

Reuses everything rather than reimplementing:
- check_company_record() / extract_acquirer_from_description() - the same
  two detectors every other script in this project calls.
- _combine_results() from test_combined_pipeline.py - the same "both agree
  -> both, disagree -> conflict logged + higher-confidence wins, single hit
  -> that path" merge logic already validated there.
- _SOURCE_FIELDS from fetch_snapshot.py - the same curated field list.

Run with: python run_full_batch.py
Resume an interrupted run: just run it again - CHECKPOINT_PATH picks up
where it left off automatically.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import check_company_record  # noqa: E402 (from name_matching)
from data_source import CompanyRecord, normalize_company_source  # noqa: E402 (from name_matching)
from description_detection_api import extract_acquirer_from_description  # noqa: E402 (from description_matching)

from fetch_snapshot import _SOURCE_FIELDS  # noqa: E402
from test_combined_pipeline import _combine_results  # noqa: E402
from s3_detection import (  # noqa: E402
    S3_BUCKET_NAME,
    S3_PREFIX,
    _s3_client,
    combined_text_blob_multi,
    extract_acquirer_from_s3_text,
    fetch_all_scrapes,
    list_pipeline_folders,
    merge_s3_result_into_db,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

DB_PATH = "combined_acquisitions_full.db"
CHECKPOINT_PATH = "full_batch_checkpoint.json"

PAGE_SIZE = 1000
MAX_WORKERS = 10  # stage 1+2: CPU-bound regex work, GIL-limited - more threads doesn't help here.
# Stage 3: pure S3 HTTP calls, no CPU-bound work and no shared ES cluster to
# overload - safe to run at much higher concurrency. Kept separate from
# MAX_WORKERS above after a real run showed stage 3 dominating total
# runtime at 10 workers (up to 6 S3 round-trips per company).
S3_MAX_WORKERS = 64
PROGRESS_EVERY = 5000

_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2

_EXISTS_QUERY = {"exists": {"field": "categoriesLLM2.keyword"}}


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
        return {"last_dwh_id": None, "processed": 0}
    with path.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    # Backward compat: checkpoints written before this fix stored a plain
    # dwhId string here, extracted from a hit's own _source. This fix
    # switches to using ES's own "sort" array for search_after instead (see
    # fetch_page()) - but dwhId.keyword is an unanalyzed keyword field, so
    # that old plain string IS exactly the value ES's sort array would have
    # held. Wrap it so an old checkpoint still resumes correctly.
    last = checkpoint.get("last_dwh_id")
    if isinstance(last, str):
        checkpoint["last_dwh_id"] = [last]
    return checkpoint


def _save_checkpoint(search_after: list[Any] | None, processed: int) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_dwh_id": search_after, "processed": processed}, f)


def fetch_page(
    es: Elasticsearch, search_after: list[Any] | None
) -> tuple[list[dict[str, Any]], list[Any] | None, int]:
    """Returns (usable company sources, ES's own "sort" value for this
    page's last hit, raw hit count). search_after pagination must always
    use that "sort" value - never anything re-extracted from a hit's own
    fields, which may be missing or shaped unexpectedly (see the module
    docstring). raw hit count is tracked separately from len(sources)
    because a hit can be skipped below without affecting how many more
    pages there are to fetch."""
    body: dict[str, Any] = {
        "size": PAGE_SIZE,
        "query": _EXISTS_QUERY,
        "sort": [{"dwhId.keyword": "asc"}],
        "_source": _SOURCE_FIELDS,
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
                "hit _id=%s has no usable _source/dwhId - skipping for detection, "
                "still advancing pagination past it",
                hit.get("_id"),
            )
            continue
        if not source.get("companyId"):
            source["companyId"] = hit.get("_id", "")
        sources.append(source)

    last_sort = hits[-1]["sort"] if hits else None
    return sources, last_sort, len(hits)


def process_company(source: dict[str, Any]) -> dict[str, Any]:
    """Never raises - failures come back as isAcquired=None instead of
    dropping the company from the batch.

    normalize_company_source() runs first and mutates `source` in place, so
    even the except branch below sees already-cleaned values (e.g.
    companyName as a plain string, not a stray ES-returned list) rather
    than the raw field that caused the failure in the first place."""
    dwh_id = source.get("dwhId", "")
    try:
        source = normalize_company_source(source)
        record = CompanyRecord(
            company_id=str(source.get("companyId") or ""),
            company_name=source.get("companyName", ""),
            raw=source,
        )
        name_match = check_company_record(record)
        description_match = extract_acquirer_from_description(record.raw.get("description"))
        combined = _combine_results(name_match, description_match, record.raw)

        return {"dwhId": dwh_id, "companyName": record.company_name, **combined}
    except Exception:
        logger.exception("dwhId %s: failed to process", dwh_id)
        return {
            "dwhId": dwh_id,
            "companyName": source.get("companyName"),
            "isAcquired": None,
            "acquirerCompany": None,
            "source": None,
            "matchedPattern": None,
            "confidenceScore": None,
            "reviewStatus": None,
        }


def _to_sqlite_value(value: Any) -> Any:
    """SQLite only binds None/int/float/str/bytes. Coerce anything else
    (e.g. a list that slipped past normalize_company_source() for some
    field this batch doesn't even normalize) to a string instead of letting
    conn.execute() raise sqlite3.ProgrammingError and abort the page's
    entire insert loop over one bad row - this is the backstop fix #3 asks
    for, independent of fix #1's normalization at the source."""
    if value is None or isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


def _create_table(conn: sqlite3.Connection) -> None:
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

    # combined_acquisitions_full.db already exists from the first full run,
    # written before reviewStatus existed - CREATE TABLE IF NOT EXISTS above
    # is a no-op against that file, so the column has to be added here too.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(combined_acquisitions)")}
    if "reviewStatus" not in existing_columns:
        conn.execute("ALTER TABLE combined_acquisitions ADD COLUMN reviewStatus TEXT")

    conn.commit()


def _run_stage3_for_page(s3, folders: list[str], dwh_id: str) -> dict[str, Any]:
    """Stage 3 (S3 web scrape) fetch + extraction only - no DB access, same
    threading discipline as s3_detection.py's own process_dwh_id (DB writes
    happen single-threaded afterward, per page)."""
    scrapes, found_in = fetch_all_scrapes(s3, S3_BUCKET_NAME, folders, dwh_id)
    if not scrapes:
        return {"dwhId": dwh_id, "companyName": None, "s3_result": None, "no_data": True}
    company_name = next((s.get("companyName") for s in scrapes if s.get("companyName")), None)
    blob = combined_text_blob_multi(scrapes)
    s3_result = extract_acquirer_from_s3_text(blob, company_name)
    return {"dwhId": dwh_id, "companyName": company_name, "s3_result": s3_result, "no_data": False}


def run_batch(es: Elasticsearch, conn: sqlite3.Connection, s3, s3_folders: list[str]) -> None:
    checkpoint = _load_checkpoint()
    search_after = checkpoint["last_dwh_id"]
    processed = checkpoint["processed"]
    stage3_outcomes: dict[str, int] = {}

    if search_after is not None:
        logger.info("Resuming from checkpoint: %d companies already processed, last dwhId=%s", processed, search_after)

    start = time.time()
    while True:
        page, last_sort, raw_hit_count = fetch_page(es, search_after)
        if raw_hit_count == 0:
            # Zero raw hits (not just zero usable sources - a page can be
            # non-empty but have every hit skipped for a missing dwhId, and
            # that's not "done", just an unusually bad page).
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_company, source) for source in page]
            results = [future.result() for future in as_completed(futures)]

        for row in results:
            dwh_id_value = _to_sqlite_value(row.get("dwhId"))
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO combined_acquisitions "
                    "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dwh_id_value,
                        _to_sqlite_value(row.get("companyName")),
                        _to_sqlite_value(row.get("isAcquired")),
                        _to_sqlite_value(row.get("acquirerCompany")),
                        _to_sqlite_value(row.get("source")),
                        _to_sqlite_value(row.get("matchedPattern")),
                        _to_sqlite_value(row.get("confidenceScore")),
                        _to_sqlite_value(row.get("reviewStatus")),
                    ),
                )
            except Exception:
                # Belt-and-suspenders: _to_sqlite_value() should already
                # prevent this, but one bad row must never abort the rest
                # of the page's writes or crash the batch - record it as an
                # error row instead of losing it silently.
                logger.exception("dwhId %s: failed to write to SQLite even after coercion", dwh_id_value)
                conn.execute(
                    "INSERT OR REPLACE INTO combined_acquisitions "
                    "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
                    "VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                    (dwh_id_value,),
                )
        conn.commit()

        # Stage 3 (S3 web scrape) for this same page - runs AFTER the stage
        # 1+2 commit above, so merge_s3_result_into_db() (which reads the
        # existing row to decide corroboration/conflict) sees this page's
        # fresh stage 1+2 verdict rather than a stale one from a prior run.
        dwh_ids_this_page = [row["dwhId"] for row in results if row.get("dwhId")]
        with ThreadPoolExecutor(max_workers=S3_MAX_WORKERS) as executor:
            futures = [
                executor.submit(_run_stage3_for_page, s3, s3_folders, dwh_id) for dwh_id in dwh_ids_this_page
            ]
            stage3_page_results = [future.result() for future in as_completed(futures)]

        for row in stage3_page_results:
            if row["no_data"]:
                stage3_outcomes["no_s3_data"] = stage3_outcomes.get("no_s3_data", 0) + 1
                continue
            outcome = merge_s3_result_into_db(conn, row["dwhId"], row["companyName"], row["s3_result"])
            stage3_outcomes[outcome] = stage3_outcomes.get(outcome, 0) + 1
        conn.commit()

        processed += len(page)
        search_after = last_sort
        _save_checkpoint(search_after, processed)

        if processed % PROGRESS_EVERY < PAGE_SIZE:
            elapsed = time.time() - start
            logger.info("Processed %d companies so far (%.1fs elapsed)", processed, elapsed)

        if raw_hit_count < PAGE_SIZE:
            break

    elapsed = time.time() - start
    logger.info("Done. Processed %d companies total in %.1fs.", processed, elapsed)

    acquired = conn.execute("SELECT COUNT(*) FROM combined_acquisitions WHERE isAcquired = 1").fetchone()[0]
    likely_acquired = conn.execute(
        "SELECT COUNT(*) FROM combined_acquisitions WHERE reviewStatus = 'likely_acquired'"
    ).fetchone()[0]
    errors = conn.execute("SELECT COUNT(*) FROM combined_acquisitions WHERE isAcquired IS NULL").fetchone()[0]
    not_acquired = processed - acquired - likely_acquired - errors
    logger.info(
        "acquired=%d likely_acquired=%d not_acquired=%d errors=%d", acquired, likely_acquired, not_acquired, errors
    )
    logger.info("stage 3 (S3): %s", stage3_outcomes)


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)
    conn = sqlite3.connect(DB_PATH)
    _create_table(conn)

    s3 = _s3_client()
    s3_folders = list_pipeline_folders(s3, S3_BUCKET_NAME, S3_PREFIX)
    logger.info("S3 pipeline folders (newest first): %s", s3_folders)

    try:
        run_batch(es, conn, s3, s3_folders)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
