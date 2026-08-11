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
DB_PATH = "acquisitions.db"
MAX_WORKERS = 5

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

# Fill in with the 10 dwhIds to test.
DWH_IDS: list[str] = [
    "b0fcc1b613b2d892eae9834345a31fb8",
    "f30bfdb5dd69c8f4a183b533fa2eb739",
    "bd3d35d96ec7c78434cfaa078a7fba7d",
    "382e49638c9f1363a26d21d664c0b704",
    "3830294b80cd3c39ab0b49d41cc002de",
    "42ef002ded27a7076ecbccd4b2398034",
    "bd1d6ea7dae2ea4257edc7d2b4120805",
    "8714fb200458bb5105caf39a045a7c19",
    "f6b305db5ef425b94911d7cc136ef52e",
    "39e2d3a104b5eb4a263d943f337478ae"
]

# Acquirer name: starts alnum, up to 50 more name-shaped chars, non-greedy.
_ACQUIRER_CAPTURE = r"(?P<acquirer>[A-Za-z0-9][\w&.,' -]{1,50}?)"
_END = r"(?=$|[.,;:)])"

_LEADING_DELIM = r"(?:[,\-–—(]\s*)*"

_ACQUISITION_PATTERNS = [
    ("part_of", re.compile(_LEADING_DELIM + r"(?:now\s+)?part\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
    ("a_x_company", re.compile(_LEADING_DELIM + r"an?\s+" + _ACQUIRER_CAPTURE + r"\s+company" + _END, re.IGNORECASE)),
    (
        "subsidiary_of",
        re.compile(r"wholly[- ]owned\s+subsidiary\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE),
    ),
    ("division_of", re.compile(r"a\s+division\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
    ("operating_unit_of", re.compile(r"operating\s+unit\s+of\s+" + _ACQUIRER_CAPTURE + _END, re.IGNORECASE)),
]


def _extract_acquirer_from_text(text: str | None) -> str | None:
    if not text:
        return None
    for _name, pattern in _ACQUISITION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group("acquirer").strip().rstrip(").,")
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


def _search_with_retry(dwh_id: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_SEARCH_ATTEMPTS + 1):
        try:
            return _es.search(
                index=INDEX,
                query={"term": {"dwhId": dwh_id}},
                source=_SOURCE_FIELDS,
                size=1,
            )
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_SEARCH_ATTEMPTS:
                time.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # type: ignore[misc]


def process_company(dwh_id: str) -> dict[str, Any]:
    """Fetch `dwh_id` and detect acquisition: acquirerCompanyName field >
    regex over companyName/linkedinProfileName > isAcquired flag > none.
    Never raises - failures return isAcquired=None instead of dropping."""
    try:
        resp = _search_with_retry(dwh_id)
        hits = resp["hits"]["hits"]
        if not hits:
            logger.error("dwhId %s: no document found", dwh_id)
            return {"dwhId": dwh_id, "companyName": None, "isAcquired": None, "acquirerCompany": None}

        source = hits[0]["_source"]
        company_name = source.get("companyName")

        acquirer = _get_nested(source, "fundingAndInvestment.acquiredBy.acquirerCompanyName")
        if acquirer:
            return {
                "dwhId": dwh_id,
                "companyName": company_name,
                "isAcquired": 1,
                "acquirerCompany": acquirer,
            }

        # Regex before the bare isAcquired flag, so a name match still fills
        # in the acquirer even when the flag alone would leave it None.
        for text in (company_name, source.get("linkedinProfileName")):
            acquirer = _extract_acquirer_from_text(text)
            if acquirer:
                return {
                    "dwhId": dwh_id,
                    "companyName": company_name,
                    "isAcquired": 1,
                    "acquirerCompany": acquirer,
                }

        if source.get("isAcquired"):
            return {
                "dwhId": dwh_id,
                "companyName": company_name,
                "isAcquired": 1,
                "acquirerCompany": None,
            }

        return {"dwhId": dwh_id, "companyName": company_name, "isAcquired": 0, "acquirerCompany": None}

    except Exception:
        logger.exception("dwhId %s: failed to process", dwh_id)
        return {"dwhId": dwh_id, "companyName": None, "isAcquired": None, "acquirerCompany": None}


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS company_acquisitions (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            isAcquired INTEGER,
            acquirerCompany TEXT
        )
        """
    )
    conn.commit()


def _print_results(conn: sqlite3.Connection, dwh_ids: list[str]) -> None:
    placeholders = ",".join("?" * len(dwh_ids))
    rows = conn.execute(
        f"SELECT dwhId, companyName, isAcquired, acquirerCompany FROM company_acquisitions "
        f"WHERE dwhId IN ({placeholders})",
        dwh_ids,
    ).fetchall()

    headers = ["dwhId", "companyName", "isAcquired", "acquirerCompany"]
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
    if not DWH_IDS:
        logger.warning("DWH_IDS is empty - fill it in with the 10 dwhIds to test.")
        return

    conn = sqlite3.connect(DB_PATH)
    _create_table(conn)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_company, dwh_id): dwh_id for dwh_id in DWH_IDS}
        for future in as_completed(futures):
            results.append(future.result())

    # Only the main thread touches SQLite - writes happen after all workers finish.
    for row in results:
        conn.execute(
            "INSERT OR REPLACE INTO company_acquisitions (dwhId, companyName, isAcquired, acquirerCompany) "
            "VALUES (?, ?, ?, ?)",
            (row["dwhId"], row["companyName"], row["isAcquired"], row["acquirerCompany"]),
        )
    conn.commit()

    _print_results(conn, DWH_IDS)
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
