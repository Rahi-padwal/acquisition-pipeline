"""Elasticsearch lookup: dwhId -> (companyName, logoS3Path).

Standalone by design (own client, own request shape) - same convention
every sibling detector in this project already follows (see s3_detection.py
and detection.py's own docstring). Duplicates rather than imports
name_matching/data_source.py's ElasticCompanyDataSource, since that class
lives in a sibling module this package must not import from.

Field confirmed against the LIVE `sellers_0726` mapping (not assumed):
`logoS3Path` exists as a top-level text field with a `.keyword` sub-field,
same shape as `dwhId`. A GET .../sellers_0726/_mapping and a sample query
via `{"exists": {"field": "logoS3Path"}}` both confirmed it, and returned
real values shaped exactly like the task description:
  https://crustdata-media.s3.us-east-2.amazonaws.com/company/<hash>.jpg

Auth/request shape (Token header, verify=False, `dwhId.keyword` for exact
terms matching) mirrors name_matching/data_source.py and
combined_matching/fetch_es_data.py - both already-established patterns in
this project, not a new convention invented here.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import urllib3
from dotenv import load_dotenv

urllib3.disable_warnings()  # cluster sits behind a proxy with a self-signed/mismatched cert

# Explicit path, not a bare load_dotenv() - this module gets imported from
# combined_matching/logo_detection.py too, where plain load_dotenv() picks
# up combined_matching/.env instead (found first via cwd/frame search) -
# that file has no ES_API_KEY (only ES_API_TOKEN, a different sibling
# script's naming), so ES_API_KEY silently ends up unset. Pointing at this
# file's own directory makes it load logo_matching/.env regardless of which
# module imports it from where - confirmed by testing the cross-import.
load_dotenv(Path(__file__).resolve().parent / ".env")

ES_HOST = os.environ.get("ES_HOST", "").rstrip("/")
ES_INDEX = os.environ.get("ES_INDEX")
ES_API_KEY = os.environ.get("ES_API_KEY")

_TIMEOUT_SECONDS = 30
# The live cluster returned a transient "index_not_found_exception" for
# sellers_0726 on some requests during testing even though the index
# unambiguously exists (71M+ docs, confirmed via _cat/indices) - a retry
# immediately succeeded every time. Not an index-name bug; looks like
# flaky routing on the cluster/proxy side. Same retry discipline
# ElasticCompanyDataSource already applies for exactly this reason.
_MAX_ATTEMPTS = 3
_RETRY_DELAY_SECONDS = 2

BATCH_SIZE = 200


@dataclass
class LogoLookup:
    dwh_id: str
    company_name: str | None
    logo_s3_path: str | None  # None if missing/null/empty in ES, or dwhId not found at all
    found_in_es: bool


def _search(body: dict) -> list[dict]:
    if not (ES_HOST and ES_INDEX and ES_API_KEY):
        raise RuntimeError("ES_HOST / ES_INDEX / ES_API_KEY must all be set (check logo_matching/.env)")

    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                f"{ES_HOST}/{ES_INDEX}/_search",
                headers={"Token": ES_API_KEY},
                json=body,
                verify=False,
                timeout=_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            last_error = f"Could not reach Elasticsearch ({ES_HOST}): {exc}"
        else:
            if resp.ok:
                return (resp.json().get("hits") or {}).get("hits") or []
            last_error = f"Elasticsearch search failed (HTTP {resp.status_code}): {resp.text[:500]}"

        if attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_DELAY_SECONDS)

    raise RuntimeError(f"{last_error} (gave up after {_MAX_ATTEMPTS} attempts)")


def _chunked(items: list[str], n: int):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def fetch_logo_lookups(dwh_ids: list[str]) -> dict[str, LogoLookup]:
    """Batched ES lookup for a list of dwhIds. Returns a dict keyed by
    dwhId - every input dwhId is present in the result, including ones
    with no ES document at all (found_in_es=False, logo_s3_path=None) so
    callers can distinguish "no ES doc" from "ES doc but no logo" if they
    want to, though both currently collapse to the same "no_logo" outcome
    in the test driver."""
    results: dict[str, LogoLookup] = {}

    for batch in _chunked(dwh_ids, BATCH_SIZE):
        body = {
            "size": len(batch),
            "_source": ["dwhId", "companyName", "logoS3Path"],
            "query": {"terms": {"dwhId.keyword": batch}},
        }
        hits = _search(body)
        for hit in hits:
            src = hit.get("_source", {})
            dwh_id = src.get("dwhId")
            if not dwh_id:
                continue
            logo_path = (src.get("logoS3Path") or "").strip() or None
            results[dwh_id] = LogoLookup(
                dwh_id=dwh_id,
                company_name=src.get("companyName"),
                logo_s3_path=logo_path,
                found_in_es=True,
            )

    for dwh_id in dwh_ids:
        if dwh_id not in results:
            results[dwh_id] = LogoLookup(dwh_id=dwh_id, company_name=None, logo_s3_path=None, found_in_es=False)

    return results
