"""CompanyDataSource backed by a local JSON snapshot instead of live
Elasticsearch - see fetch_snapshot.py for how that snapshot gets built.

Implements the same CompanyDataSource Protocol defined in name_matching/
data_source.py, so check_company_record() / extract_acquirer_from_description()
run against it exactly as they do against ElasticCompanyDataSource - no
detection logic changes needed. Useful for reproducible test runs: a live
random sample picks different companies every run, this replays the exact
same batch every time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_NAME_MATCHING_PATH = str(_ROOT / "name_matching")
if _NAME_MATCHING_PATH not in sys.path:
    sys.path.insert(0, _NAME_MATCHING_PATH)

from data_source import CompanyRecord  # noqa: E402 (from name_matching)


class JsonCompanyDataSource:
    """Loads a JSON snapshot (list of company objects, see fetch_snapshot.py
    for the exact shape) into memory once at construction, then serves
    get_by_id / get_by_dwh_id / search_by_name / search_by_domain from
    that in-memory copy - no network calls."""

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        with self.json_path.open("r", encoding="utf-8") as f:
            companies: list[dict[str, Any]] = json.load(f)

        self._by_id: dict[str, CompanyRecord] = {}
        self._by_dwh_id: dict[str, CompanyRecord] = {}
        self._records: list[CompanyRecord] = []

        for company in companies:
            company_id = str(company.get("companyId") or "")
            record = CompanyRecord(
                company_id=company_id,
                company_name=company.get("companyName", ""),
                raw=company,
            )
            self._records.append(record)
            if company_id:
                self._by_id[company_id] = record
            dwh_id = company.get("dwhId")
            if dwh_id:
                self._by_dwh_id[str(dwh_id)] = record

    def get_by_id(self, company_id: str) -> CompanyRecord | None:
        return self._by_id.get(str(company_id))

    def get_by_dwh_id(self, dwh_id: str) -> CompanyRecord | None:
        return self._by_dwh_id.get(str(dwh_id))

    def search_by_name(self, name: str) -> list[CompanyRecord]:
        needle = name.strip().lower()
        return [r for r in self._records if needle in r.company_name.lower()]

    def search_by_domain(self, domain: str) -> list[CompanyRecord]:
        needle = domain.strip().lower()
        return [r for r in self._records if needle in (r.raw.get("companyWebsiteDomain") or "").lower()]

    def all_dwh_ids(self) -> list[str]:
        """Every dwhId in the snapshot, in file order - lets a caller
        replay detection over the whole snapshot without needing its own
        copy of which companies it contains."""
        return list(self._by_dwh_id.keys())
