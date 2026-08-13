"""Smoke test: run acquisition detection over a small hardcoded list of
dwhIds pulled from Elasticsearch, store results in SQLite, print them.

Run with: python test_acquisition_pipeline.py
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()  # picks up ES_HOST / ES_API_KEY from .env, if present

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Gateway only accepts a custom `Token` header; cert is self-signed/mismatched.
ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
INDEX = "sellers_0726"
DB_PATH = "test_acquisitions.db"
MAX_WORKERS = 10

# Cluster intermittently 404s transiently - retry before giving up.
_MAX_SEARCH_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2

_SOURCE_FIELDS = [
    "dwhId",
    "companyName",
    "isAcquired",
    "fundingAndInvestment.acquiredBy.acquirerCompanyName",
    "fundingAndInvestment.acquiredBy.acquireeCompanyName",
    "linkedinProfileName",
]

DWH_COUNT = 10

# Acquirer name: starts alnum, up to 50 more name-shaped chars, non-greedy.
_ACQUIRER_CAPTURE = r"(?P<acquirer>[A-Za-z0-9][\w&.,' -]{1,50}?)"
_END = r"(?=$|[.,;:)])"

_LEADING_DELIM = r"(?:[,\-–—(]\s*)*"

_ACQUISITION_PATTERNS = [
    ("part_of", re.compile(_LEADING_DELIM + r"(?:now\s+)?part\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
    ("a_x_company", re.compile(_LEADING_DELIM + r"\ban?\s+" + _ACQUIRER_CAPTURE + r"\s+company" + _END, re.IGNORECASE)),
    (
        "subsidiary_of",
        re.compile(r"wholly[- ]owned\s+subsidiary\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE),
    ),
    ("division_of", re.compile(r"\ba\s+division\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
    ("operating_unit_of", re.compile(r"operating\s+unit\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
]


def _extract_acquirer_from_text(text: str | None) -> tuple[str, str] | None:
    """Returns (acquirer, tag) for the first pattern that matches, or None."""
    if not text:
        return None
    for tag, pattern in _ACQUISITION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("acquirer").strip().rstrip(").,"), tag
    return None


def _get_nested(source: dict[str, Any], dotted_path: str) -> Any:
    value: Any = source
    for part in dotted_path.split("."):
        if isinstance(value, list):
            value = value[0] if value else {}
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _es_search_with_retry(**kwargs: Any) -> dict[str, Any]:
    """Retries any `_es.search(...)` call - the cluster intermittently 404s
    transiently (see module docstring / pipeline.py for the same pattern)."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            return _es.search(index=INDEX, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_SEARCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # type: ignore[misc]


def _search_with_retry(dwh_id: str) -> dict[str, Any]:
    return _es_search_with_retry(
        query={"term": {"dwhId": dwh_id}},
        source=_SOURCE_FIELDS,
        size=1,
    )


def get_dwh_ids(size: int) -> list[str]:
    """Random sample of `size` dwhIds, restricted to companies that have a
    categoriesLLM2 value populated (via random_score, not the first N by
    index order)."""
    resp = _es_search_with_retry(
        query={
            "function_score": {
                "query": {"exists": {"field": "categoriesLLM2.keyword"}},
                "random_score": {},
            }
        },
        source=["dwhId"],
        size=size,
    )
    return [
        hit["_source"]["dwhId"]
        for hit in resp["hits"]["hits"]
        if hit["_source"].get("dwhId")
    ]


def process_company(dwh_id: str) -> dict[str, Any]:
    """Fetch `dwh_id` and detect acquisition: acquirerCompanyName field >
    regex over companyName/linkedinProfileName > isAcquired flag > none.
    `tag` records which of those signals produced the result, for auditing.
    Never raises - failures return isAcquired=None instead of dropping."""
    try:
        resp = _search_with_retry(dwh_id)
        hits = resp["hits"]["hits"]
        if not hits:
            logger.error("dwhId %s: no document found", dwh_id)
            return {"dwhId": dwh_id, "companyName": None, "isAcquired": None, "acquirerCompany": None, "tag": None}

        source = hits[0]["_source"]
        company_name = source.get("companyName")

        acquirer = _get_nested(source, "fundingAndInvestment.acquiredBy.acquirerCompanyName")
        if acquirer:
            return {
                "dwhId": dwh_id,
                "companyName": company_name,
                "isAcquired": 1,
                "acquirerCompany": acquirer,
                "tag": "acquirerCompanyName",
            }

        # Regex before the bare isAcquired flag, so a name match still fills
        # in the acquirer even when the flag alone would leave it None.
        for text in (company_name, source.get("linkedinProfileName")):
            match = _extract_acquirer_from_text(text)
            if match:
                acquirer, tag = match
                return {
                    "dwhId": dwh_id,
                    "companyName": company_name,
                    "isAcquired": 1,
                    "acquirerCompany": acquirer,
                    "tag": tag,
                }

        if source.get("isAcquired"):
            return {
                "dwhId": dwh_id,
                "companyName": company_name,
                "isAcquired": 1,
                "acquirerCompany": None,
                "tag": "isAcquired_flag",
            }

        return {"dwhId": dwh_id, "companyName": company_name, "isAcquired": 0, "acquirerCompany": None, "tag": None}

    except Exception:
        logger.exception("dwhId %s: failed to process", dwh_id)
        return {"dwhId": dwh_id, "companyName": None, "isAcquired": None, "acquirerCompany": None, "tag": None}


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS company_acquisitions (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            isAcquired INTEGER,
            acquirerCompany TEXT,
            tag TEXT
        )
        """
    )

    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(company_acquisitions)")}
    if "tag" not in existing_columns:
        conn.execute("ALTER TABLE company_acquisitions ADD COLUMN tag TEXT")

    conn.commit()


def _print_results(conn: sqlite3.Connection, dwh_ids: list[str]) -> None:
    placeholders = ",".join("?" * len(dwh_ids))
    rows = conn.execute(
        f"SELECT dwhId, companyName, isAcquired, acquirerCompany, tag FROM company_acquisitions "
        f"WHERE dwhId IN ({placeholders})",
        dwh_ids,
    ).fetchall()

    headers = ["dwhId", "companyName", "isAcquired", "acquirerCompany", "tag"]
    widths = [
        max(len(headers[i]), *(len(str(row[i])) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(values: tuple) -> str:
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    print(fmt_row(tuple(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    dwh_ids = get_dwh_ids(DWH_COUNT)
    if not dwh_ids:
        logger.warning("No dwhIds returned from Elasticsearch.")
        return

    conn = sqlite3.connect(DB_PATH)
    _create_table(conn)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_company, dwh_id): dwh_id for dwh_id in dwh_ids}
        for future in as_completed(futures):
            results.append(future.result())

    # Only the main thread touches SQLite - writes happen after all workers finish.
    for row in results:
        conn.execute(
            "INSERT OR REPLACE INTO company_acquisitions (dwhId, companyName, isAcquired, acquirerCompany, tag) "
            "VALUES (?, ?, ?, ?, ?)",
            (row["dwhId"], row["companyName"], row["isAcquired"], row["acquirerCompany"], row["tag"]),
        )
    conn.commit()

    _print_results(conn, dwh_ids)
    conn.close()


if not ES_API_KEY:
    raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

_es = Elasticsearch(
    ES_HOST,
    headers={"Token": ES_API_KEY},
    verify_certs=False,
    ssl_show_warn=False,
)

if __name__ == "__main__":
    main()
