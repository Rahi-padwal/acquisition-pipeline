"""Run the combined detection pipeline (companyName-based + description-
based) over EVERY company in sellers_0726 that has categoriesLLM2
populated - not a sample, the whole ~768k-document set (confirmed via a
_count query - see the conversation this was built from).

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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

DB_PATH = "combined_acquisitions_full.db"
CHECKPOINT_PATH = "full_batch_checkpoint.json"

PAGE_SIZE = 1000
MAX_WORKERS = 10
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
            confidenceScore REAL
        )
        """
    )
    conn.commit()


def run_batch(es: Elasticsearch, conn: sqlite3.Connection) -> None:
    checkpoint = _load_checkpoint()
    search_after = checkpoint["last_dwh_id"]
    processed = checkpoint["processed"]

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
                    "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        dwh_id_value,
                        _to_sqlite_value(row.get("companyName")),
                        _to_sqlite_value(row.get("isAcquired")),
                        _to_sqlite_value(row.get("acquirerCompany")),
                        _to_sqlite_value(row.get("source")),
                        _to_sqlite_value(row.get("matchedPattern")),
                        _to_sqlite_value(row.get("confidenceScore")),
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
                    "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore) "
                    "VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL)",
                    (dwh_id_value,),
                )
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
    errors = conn.execute("SELECT COUNT(*) FROM combined_acquisitions WHERE isAcquired IS NULL").fetchone()[0]
    logger.info("acquired=%d not_acquired=%d errors=%d", acquired, processed - acquired - errors, errors)


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
