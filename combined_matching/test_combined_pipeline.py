"""Smoke test: run BOTH detection paths (companyName-based + description-
based) together on a genuinely random sample of companies, store combined
results in SQLite, print them.

Reuses the two existing detection modules as-is:
- check_name_signals.check_company_record() (name_matching) - structured ES
  fields, companyName regex, funding-round parentheticals, already scored.
- description_detection_api.extract_acquirer_from_description()
  (description_matching) - phrase patterns over the `description` field,
  already scored.

The merge logic below (agreement -> "both", disagreement -> conflict
logged + higher-confidence one kept, single hit -> that path) is specific
to THIS script and intentionally different from combined_matching/
combined_check.py's own priority rules (which always prefer the name-based
result and never label a source "both") - see _combine_results().

Run with: python test_combined_pipeline.py
Or replay a saved snapshot (see fetch_snapshot.py) instead of live ES:
    python test_combined_pipeline.py snapshot.json
"""

from __future__ import annotations

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

from check_name_signals import _company_identity_match, check_company_record  # noqa: E402 (from name_matching)
from combined_check import _apply_structured_field_confidence_boost  # noqa: E402
from data_source import CompanyDataSource, ElasticCompanyDataSource  # noqa: E402 (from name_matching)
from description_detection_api import extract_acquirer_from_description  # noqa: E402 (from description_matching)
from json_data_source import JsonCompanyDataSource  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Gateway only accepts a custom `Token` header; cert is self-signed/mismatched.
ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")
DB_PATH = "test_combined_pipeline.db"

SAMPLE_SIZE = 100
MAX_WORKERS = 10

# Cluster intermittently 404s transiently - retry before giving up (see
# pipeline.py / test_acquisition_pipeline.py for the same pattern).
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


def get_random_dwh_ids(es: Elasticsearch, size: int) -> list[str]:
    """Random sample of `size` dwhIds via a random_score function query
    (same approach as pipeline.py's fetch_random_sample(), scaled down to a
    single page since size is small here), restricted to companies that
    have a categoriesLLM2 value populated - same filter as name_matching/
    test_acquisition_pipeline.py's get_dwh_ids()."""
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
            "_source": ["dwhId"],
        },
    )
    return [hit["_source"]["dwhId"] for hit in resp["hits"]["hits"] if hit["_source"].get("dwhId")]


def _combine_results(
    name_match: dict[str, Any] | None,
    description_match: dict[str, Any] | None,
    raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge the two detectors' outputs into one verdict:
    - both agree on the acquirer -> source="both", higher confidence score kept
    - only one path has a match -> that path's result, source="company_name"/"description"
    - both match but disagree -> conflict logged, higher-confidence candidate
      kept as the recorded acquirer (never silently dropped - the other
      candidate is visible in the warning log)
    - neither matches -> no acquirer at all

    `raw` (the company's raw ES source) is optional only for backward
    compatibility with old call sites - pass it so the final
    _apply_structured_field_confidence_boost() post-processing step below can
    see fundingAndInvestment.acquiredBy.acquirerCompanyName; without it, that
    step is skipped.
    """
    description_acquirer = (
        description_match["acquirerCompany"]
        if description_match is not None and description_match["acquirerCompany"] is not None
        else None
    )

    if name_match is None and description_acquirer is None:
        return {"isAcquired": 0, "acquirerCompany": None, "source": None, "matchedPattern": None, "confidenceScore": None}

    if name_match is not None and description_acquirer is None:
        matched_pattern = name_match["matched_pattern"]
        score = name_match.get("confidence_score")
        # Uncorroborated cap only exists on the name-match side (see
        # check_name_signals._score_match), so it's the only notes value the
        # boost step needs to know about here.
        notes = name_match.get("confidence_notes") if matched_pattern == "structured_field" else None
        # Skipped when the match itself IS structured_field - see the
        # matching guard/comment in combined_check.py's run_combined_check().
        # Boosting a structured_field match using the structured field is
        # circular (the field being "populated" is guaranteed by definition
        # here) and was spamming a meaningless warning on every uncorroborated
        # structured_field row in a full-batch run.
        if raw is not None and matched_pattern != "structured_field":
            score, _ = _apply_structured_field_confidence_boost(score, notes, raw)
        return {
            "isAcquired": 1,
            "acquirerCompany": name_match["candidate_name"],
            "source": "company_name",
            "matchedPattern": matched_pattern,
            "confidenceScore": score,
        }

    if name_match is None and description_acquirer is not None:
        score = description_match.get("confidenceScore")
        if raw is not None:
            score, _ = _apply_structured_field_confidence_boost(score, None, raw)
        return {
            "isAcquired": 1,
            "acquirerCompany": description_acquirer,
            "source": "description",
            "matchedPattern": description_match["matchedPattern"],
            "confidenceScore": score,
        }

    # Both paths found a candidate - compare them.
    name_score = name_match.get("confidence_score") or 0.0
    description_score = description_match.get("confidenceScore") or 0.0
    name_wins = name_score >= description_score

    if _company_identity_match(name_match["candidate_name"], description_acquirer):
        acquirer = name_match["candidate_name"] if name_wins else description_acquirer
        matched_pattern = name_match["matched_pattern"] if name_wins else description_match["matchedPattern"]
        # Agreement is itself a confidence signal - report the higher of the
        # two base scores rather than either alone (mirrors the +bonus idea
        # in combined_matching/combined_check.py's corroboration modifier,
        # kept simple here since this script's own priority rules already
        # differ from that module's - see the module docstring).
        confidence_score = max(name_score, description_score)
        notes = name_match.get("confidence_notes") if matched_pattern == "structured_field" else None
        if raw is not None and matched_pattern != "structured_field":
            confidence_score, _ = _apply_structured_field_confidence_boost(confidence_score, notes, raw)
        return {
            "isAcquired": 1,
            "acquirerCompany": acquirer,
            "source": "both",
            "matchedPattern": matched_pattern,
            "confidenceScore": confidence_score,
        }

    if name_wins:
        primary_acquirer, primary_pattern, primary_source, primary_score = (
            name_match["candidate_name"],
            name_match["matched_pattern"],
            "company_name",
            name_score,
        )
    else:
        primary_acquirer, primary_pattern, primary_source, primary_score = (
            description_acquirer,
            description_match["matchedPattern"],
            "description",
            description_score,
        )

    logger.warning(
        "Conflict: companyName path says %r, description path says %r - "
        "keeping %r as primary (higher confidence score)",
        name_match["candidate_name"],
        description_acquirer,
        primary_acquirer,
    )
    if raw is not None and primary_pattern != "structured_field":
        # primary_pattern != "structured_field" here (guarded above), so
        # there's no structured_field_uncorroborated note to pass through.
        primary_score, _ = _apply_structured_field_confidence_boost(primary_score, None, raw)
    return {
        "isAcquired": 1,
        "acquirerCompany": primary_acquirer,
        "source": primary_source,
        "matchedPattern": primary_pattern,
        "confidenceScore": primary_score,
    }


def process_company(dwh_id: str, data_source: CompanyDataSource) -> dict[str, Any]:
    """Run both detection paths on `dwh_id` and merge the results. Never
    raises - failures come back as isAcquired=None instead of dropping."""
    try:
        record = data_source.get_by_dwh_id(dwh_id)
        if record is None:
            logger.error("dwhId %s: no document found", dwh_id)
            return {
                "dwhId": dwh_id,
                "companyName": None,
                "isAcquired": None,
                "acquirerCompany": None,
                "source": None,
                "matchedPattern": None,
                "confidenceScore": None,
            }

        name_match = check_company_record(record)
        description_match = extract_acquirer_from_description(record.raw.get("description"))
        combined = _combine_results(name_match, description_match, record.raw)

        return {
            "dwhId": dwh_id,
            "companyName": record.company_name,
            **combined,
        }
    except Exception:
        logger.exception("dwhId %s: failed to process", dwh_id)
        return {
            "dwhId": dwh_id,
            "companyName": None,
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

    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(combined_acquisitions)")}
    if "confidenceScore" not in existing_columns:
        conn.execute("ALTER TABLE combined_acquisitions ADD COLUMN confidenceScore REAL")

    conn.commit()


def _safe_str(value: Any) -> str:
    """str(value), with any character the current console encoding can't
    display swapped for '?' instead of crashing the whole run. Windows
    terminals commonly default to cp1252, and companyName routinely has
    characters (curly quotes, non-Latin scripts, etc.) outside that set."""
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


def _print_results(conn: sqlite3.Connection, dwh_ids: list[str]) -> None:
    placeholders = ",".join("?" * len(dwh_ids))
    rows = conn.execute(
        f"SELECT dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore "
        f"FROM combined_acquisitions WHERE dwhId IN ({placeholders})",
        dwh_ids,
    ).fetchall()

    headers = ["dwhId", "companyName", "isAcquired", "acquirerCompany", "source", "matchedPattern", "confidenceScore"]
    widths = [
        max(len(headers[i]), *(len(_safe_str(row[i])) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(values: tuple) -> str:
        return " | ".join(_safe_str(v).ljust(widths[i]) for i, v in enumerate(values))

    print(fmt_row(tuple(headers)))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    # Replay against a saved JSON snapshot (see fetch_snapshot.py) instead
    # of live ES when one is given - via `python test_combined_pipeline.py
    # <snapshot.json>` or the SNAPSHOT_PATH env var. Same detection code
    # either way; only where the companies come from changes.
    snapshot_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SNAPSHOT_PATH")

    data_source: CompanyDataSource
    if snapshot_path:
        logger.info("Replaying against JSON snapshot: %s", snapshot_path)
        data_source = JsonCompanyDataSource(snapshot_path)
        dwh_ids = data_source.all_dwh_ids()
    else:
        if not ES_API_KEY:
            raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")
        es = Elasticsearch(ES_HOST, headers={"Token": ES_API_KEY}, verify_certs=False, ssl_show_warn=False)
        data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)
        dwh_ids = get_random_dwh_ids(es, SAMPLE_SIZE)

    if not dwh_ids:
        logger.warning("No dwhIds found.")
        return

    conn = sqlite3.connect(DB_PATH)
    _create_table(conn)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_company, dwh_id, data_source): dwh_id for dwh_id in dwh_ids}
        for future in as_completed(futures):
            results.append(future.result())

    # Only the main thread touches SQLite - writes happen after all workers finish.
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

    _print_results(conn, dwh_ids)
    conn.close()


if __name__ == "__main__":
    main()
