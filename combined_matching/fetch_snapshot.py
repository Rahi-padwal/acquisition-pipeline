"""Fetch a random, categoriesLLM2-filtered sample of companies from
Elasticsearch and save the fields detection needs to a JSON file - see
json_data_source.py for how that file gets replayed back into
check_company_record() / extract_acquirer_from_description() without
touching ES again.

Run with: python fetch_snapshot.py [size] [output_path]
Defaults: size=10, output_path=snapshot.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
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

DEFAULT_SIZE = 10
DEFAULT_OUTPUT = "snapshot.json"

_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2

# Only what check_company_record() / extract_acquirer_from_description() /
# the confidence-scoring modifiers actually read - see the approved plan
# for why each field is here.
_SOURCE_FIELDS = [
    "dwhId",
    "companyId",
    "companyName",
    "description",
    "isAcquired",
    "fundingAndInvestment.acquiredBy.acquirerCompanyName",
    "fundingAndInvestment.acquiredBy.acquireeCompanyName",
    "fundingAndInvestment.acquiredBy.acquirerCompanyId",
    "fundingAndInvestment.fundingMilestonesTimeseries.fundingRound",
    "linkedinProfileName",
    "companyWebsiteDomain",
    "categoriesLLM2",
]


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


def fetch_snapshot(es: Elasticsearch, size: int) -> list[dict[str, Any]]:
    """Random sample of `size` companies, restricted to ones with a
    categoriesLLM2 value populated - same filter used by the live test
    scripts' random sampling, so a snapshot is a faithful stand-in."""
    seed = os.urandom(4).hex()
    resp = _search_with_retry(
        es,
        {
            "size": size,
            "query": {
                "function_score": {
                    "query": {"exists": {"field": "categoriesLLM2.keyword"}},
                    "random_score": {"seed": seed, "field": "_seq_no"},
                }
            },
            "sort": ["_score"],
            "_source": _SOURCE_FIELDS,
        },
    )

    companies: list[dict[str, Any]] = []
    for hit in resp["hits"]["hits"]:
        source = hit.get("_source", {})
        # Same fallback ElasticCompanyDataSource._to_record() uses - a
        # missing/null companyId in _source falls back to the doc's _id.
        if not source.get("companyId"):
            source["companyId"] = hit.get("_id", "")
        companies.append(source)
    return companies


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SIZE
    output_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)

    logger.info("Fetching %d companies (categoriesLLM2 populated, random sample)...", size)
    companies = fetch_snapshot(es, size)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(companies, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d companies to %s", len(companies), output_path)


if __name__ == "__main__":
    main()
