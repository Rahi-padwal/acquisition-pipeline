"""End-to-end acquisition-detection pipeline.

Step 1: pull a random sample of companies (dwhId + companyName) from
        Elasticsearch (crustdata) into the `companies` table.
Step 2: run acquisition detection over every dwhId just fetched and store
        results in the `company_acquisitions` table.

Run with: python pipeline.py
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

from check_name_signals import run_dwh_id_check
from data_source import ElasticCompanyDataSource

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")
DB_PATH = "acquisitions.db"

SAMPLE_SIZE = 1000
PAGE_SIZE = 200
MAX_WORKERS = 10
PROGRESS_EVERY = 50

# Cluster intermittently 404s transiently - retry before giving up.
_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


# --- Step 1: fetch dwhIds from Elasticsearch -------------------------------


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS company_acquisitions (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            isAcquired INTEGER,
            acquirerName TEXT,
            rawName TEXT
        )
        """
    )
    conn.commit()


def _search_with_retry(es: Elasticsearch, body: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            return es.search(index=ES_INDEX, body=body)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_SEARCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # type: ignore[misc]


def fetch_random_sample(es: Elasticsearch, size: int) -> list[dict]:
    """Random sample via a random_score function query, paginated with
    search_after on a random seed field so no single request needs `size`
    larger than PAGE_SIZE."""
    seed = os.urandom(4).hex()
    hits: list[dict] = []
    search_after = None

    while len(hits) < size:
        body: dict = {
            "size": min(PAGE_SIZE, size - len(hits)),
            "query": {
                "function_score": {
                    "query": {"match_all": {}},
                    "random_score": {"seed": seed, "field": "_seq_no"},
                }
            },
            "sort": ["_score", {"_seq_no": "asc"}],
            "_source": ["dwhId", "companyName"],
        }
        if search_after is not None:
            body["search_after"] = search_after

        resp = _search_with_retry(es, body)
        page = resp["hits"]["hits"]
        if not page:
            break

        hits.extend(page)
        search_after = page[-1]["sort"]

    return hits[:size]


def fetch_dwh_ids(conn: sqlite3.Connection, es: Elasticsearch) -> list[str]:
    logger.info("Fetching %d random companies from '%s'...", SAMPLE_SIZE, ES_INDEX)
    hits = fetch_random_sample(es, SAMPLE_SIZE)

    rows = []
    skipped = 0
    for hit in hits:
        source = hit["_source"]
        dwh_id = source.get("dwhId")
        if not dwh_id:
            skipped += 1
            continue
        rows.append((dwh_id, source.get("companyName")))

    conn.executemany("INSERT OR REPLACE INTO companies (dwhId, companyName) VALUES (?, ?)", rows)
    conn.commit()

    if skipped:
        logger.warning("Skipped %d hits with no dwhId.", skipped)
    logger.info("Fetched %d companies.", len(rows))
    return [dwh_id for dwh_id, _ in rows]


# --- Step 2: run acquisition detection over those dwhIds -------------------


def process_company(dwh_id: str, data_source: ElasticCompanyDataSource) -> dict[str, Any]:
    """Never raises - failures come back as status="error" instead of dropping."""
    try:
        result = run_dwh_id_check(dwh_id, data_source)
        evidence = result["evidence"]
        return {
            "dwhId": dwh_id,
            "companyName": result["company_name"],
            "isAcquired": int(result["is_acquired"]),
            "acquirerName": result["likely_acquirer"],
            "rawName": evidence[0]["raw_name"] if evidence else None,
            "status": result["status"],
        }
    except Exception:
        logger.exception("dwhId %s: failed to process", dwh_id)
        return {
            "dwhId": dwh_id,
            "companyName": None,
            "isAcquired": None,
            "acquirerName": None,
            "rawName": None,
            "status": "error",
        }


def run_acquisition_checks(conn: sqlite3.Connection, dwh_ids: list[str]) -> None:
    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)

    logger.info("Checking %d companies for acquisitions with %d workers...", len(dwh_ids), MAX_WORKERS)

    start = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_company, dwh_id, data_source): dwh_id for dwh_id in dwh_ids}
        for i, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if i % PROGRESS_EVERY == 0 or i == len(dwh_ids):
                logger.info("Processed %d/%d", i, len(dwh_ids))

    # Only the main thread touches SQLite - writes happen after all workers finish.
    for row in results:
        conn.execute(
            "INSERT OR REPLACE INTO company_acquisitions "
            "(dwhId, companyName, isAcquired, acquirerName, rawName) VALUES (?, ?, ?, ?, ?)",
            (row["dwhId"], row["companyName"], row["isAcquired"], row["acquirerName"], row["rawName"]),
        )
    conn.commit()

    elapsed = time.time() - start
    acquired = sum(1 for r in results if r["isAcquired"] == 1)
    not_found = sum(1 for r in results if r["status"] == "not_found")
    errors = sum(1 for r in results if r["status"] == "error")
    logger.info(
        "Done in %.1fs. acquired=%d not_acquired=%d not_found=%d errors=%d",
        elapsed,
        acquired,
        len(results) - acquired - not_found - errors,
        not_found,
        errors,
    )


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)

    conn = sqlite3.connect(DB_PATH)
    _create_tables(conn)

    dwh_ids = fetch_dwh_ids(conn, es)
    if not dwh_ids:
        logger.warning("No dwhIds fetched - nothing to check.")
        conn.close()
        return

    run_acquisition_checks(conn, dwh_ids)
    conn.close()


if __name__ == "__main__":
    main()
