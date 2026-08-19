"""Fetch companyWebsiteDomain for every dwhId in acquired_companies_final.csv
and write a new CSV with the domain attached.

Reuses the same Elasticsearch client setup as fetch_snapshot.py/
run_full_batch.py (Token-header auth, ES_HOST/ES_INDEX/ES_API_KEY from .env,
the same _search_with_retry retry pattern) - the only new logic here is a
batched `terms` lookup (many dwhIds per request) instead of one request per
company, so ~thousands of companies don't mean thousands of round trips.

Run with: python fetch_domains_for_acquired.py
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

INPUT_PATH = "acquired_companies_final.csv"
OUTPUT_PATH = "domains_for_acquired.csv"
BATCH_SIZE = 500

# Cluster intermittently 404s transiently - retry before giving up (same
# pattern as fetch_snapshot.py / run_full_batch.py).
_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2


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


def _read_input_rows(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fetch_domains_batch(es: Elasticsearch, dwh_ids: list[str]) -> dict[str, str]:
    """One batch: dwhId -> companyWebsiteDomain for every id ES actually has
    a hit for. Ids with no hit, or a hit with no domain, are simply absent
    from the returned dict - the caller fills those in blank, not an error."""
    resp = _search_with_retry(
        es,
        {
            "size": len(dwh_ids),
            "query": {"terms": {"dwhId": dwh_ids}},
            "_source": ["dwhId", "companyWebsiteDomain"],
        },
    )
    domains: dict[str, str] = {}
    for hit in resp["hits"]["hits"]:
        source = hit.get("_source") or {}
        dwh_id = source.get("dwhId")
        if dwh_id:
            domains[dwh_id] = source.get("companyWebsiteDomain") or ""
    return domains


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    rows = _read_input_rows(INPUT_PATH)
    logger.info("Read %d rows from %s", len(rows), INPUT_PATH)

    es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)

    dwh_ids = [row["dwhId"] for row in rows]
    domain_by_dwh_id: dict[str, str] = {}

    for batch_start in range(0, len(dwh_ids), BATCH_SIZE):
        batch = dwh_ids[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(dwh_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        try:
            domain_by_dwh_id.update(fetch_domains_batch(es, batch))
        except Exception:
            # Never crash the whole run over one bad batch - those dwhIds
            # just end up with a blank domain in the output, same as any
            # dwhId ES has no hit for.
            logger.exception(
                "Batch %d/%d (%d dwhIds) failed after retries - leaving domains blank for this batch",
                batch_num, total_batches, len(batch),
            )
        logger.info("Processed batch %d/%d (%d dwhIds)", batch_num, total_batches, len(batch))

    with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dwhId", "companyName", "companyWebsiteDomain"])
        for row in rows:
            writer.writerow([row["dwhId"], row["companyName"], domain_by_dwh_id.get(row["dwhId"], "")])

    with_domain = sum(1 for v in domain_by_dwh_id.values() if v)
    logger.info("Wrote %d rows to %s (%d with a non-blank domain)", len(rows), OUTPUT_PATH, with_domain)


if __name__ == "__main__":
    main()
