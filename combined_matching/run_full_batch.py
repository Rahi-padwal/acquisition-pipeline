"""Run the combined detection pipeline (companyName-based + description-
based) over EVERY company in sellers_0726 that has categoriesLLM2
populated - not a sample, the whole ~768k-document set (confirmed via a
_count query - see the conversation this was built from).

Unlike test_combined_pipeline.py's random_score sampling, this paginates
through the FULL matching set via search_after, sorted by dwhId.keyword
(dwhId itself is mapped `text`, not sortable directly - ES rejects sorting
on analyzed text fields; dwhId.keyword is the companion keyword sub-field,
same pattern as companyName.keyword / companyWebsiteDomain.keyword
elsewhere in this project). Checkpointed: the last dwhId processed is written to
CHECKPOINT_PATH after every page, so a crash or Ctrl+C can be resumed by
just running this again rather than restarting from page 1 - relevant at
this scale, since ~768 page-fetches means the cluster's known intermittent
404 flakiness (see _search_with_retry() elsewhere in this project) is very
likely to be hit at least once over a full run.

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
from data_source import CompanyRecord  # noqa: E402 (from name_matching)
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
        return json.load(f)


def _save_checkpoint(last_dwh_id: str | None, processed: int) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_dwh_id": last_dwh_id, "processed": processed}, f)


def fetch_page(es: Elasticsearch, search_after: str | None) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "size": PAGE_SIZE,
        "query": _EXISTS_QUERY,
        "sort": [{"dwhId.keyword": "asc"}],
        "_source": _SOURCE_FIELDS,
    }
    if search_after is not None:
        body["search_after"] = [search_after]

    resp = _search_with_retry(es, body)
    hits = resp["hits"]["hits"]

    sources = []
    for hit in hits:
        source = hit.get("_source", {})
        if not source.get("companyId"):
            source["companyId"] = hit.get("_id", "")
        sources.append(source)
    return sources


def process_company(source: dict[str, Any]) -> dict[str, Any]:
    """Never raises - failures come back as isAcquired=None instead of
    dropping the company from the batch."""
    dwh_id = source.get("dwhId", "")
    try:
        record = CompanyRecord(
            company_id=str(source.get("companyId") or ""),
            company_name=source.get("companyName", ""),
            raw=source,
        )
        name_match = check_company_record(record)
        description_match = extract_acquirer_from_description(record.raw.get("description"))
        combined = _combine_results(name_match, description_match)

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
        page = fetch_page(es, search_after)
        if not page:
            break

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_company, source) for source in page]
            results = [future.result() for future in as_completed(futures)]

        for row in results:
            conn.execute(
                "INSERT OR REPLACE INTO combined_acquisitions "
                "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    row["dwhId"],
                    row["companyName"],
                    row["isAcquired"],
                    row["acquirerCompany"],
                    row["source"],
                    row["matchedPattern"],
                    row["confidenceScore"],
                ),
            )
        conn.commit()

        processed += len(page)
        search_after = page[-1]["dwhId"]
        _save_checkpoint(search_after, processed)

        if processed % PROGRESS_EVERY < PAGE_SIZE:
            elapsed = time.time() - start
            logger.info("Processed %d companies so far (%.1fs elapsed)", processed, elapsed)

        if len(page) < PAGE_SIZE:
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
