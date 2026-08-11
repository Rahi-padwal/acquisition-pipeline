"""Data access for the name-matching pipeline.

Defines the CompanyDataSource interface plus two implementations:
StubCompanyDataSource (in-memory, for tests) and ElasticCompanyDataSource
(real ES via ES_HOST/ES_INDEX/ES_API_KEY). check_name_signals.py depends
only on the interface, so swapping implementations is a one-line change.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Protocol

import requests
import urllib3

urllib3.disable_warnings()  # cluster sits behind a proxy with a self-signed/mismatched cert


@dataclass
class CompanyRecord:
    """A company record as read from the company Elastic index. Only the
    couple fields this pipeline uses are pulled out; everything else is
    kept in `raw`."""

    company_id: str
    company_name: str
    raw: dict = field(default_factory=dict)


class CompanyDataSource(Protocol):
    """What check_name_signals.py needs from wherever company records live."""

    def get_by_id(self, company_id: str) -> CompanyRecord | None:
        """Look up a single company by its companyId."""
        ...

    def get_by_dwh_id(self, dwh_id: str) -> CompanyRecord | None:
        """Look up a single company by its dwhId."""
        ...

    def search_by_name(self, name: str) -> list[CompanyRecord]:
        """Look up companies by (fuzzy/exact) companyName match."""
        ...

    def search_by_domain(self, domain: str) -> list[CompanyRecord]:
        """Look up companies by companyWebsiteDomain match."""
        ...


class StubCompanyDataSource:
    """In-memory CompanyDataSource - no Elasticsearch required."""

    def __init__(self, records: list[CompanyRecord] | None = None):
        self._by_id: dict[str, CompanyRecord] = {}
        self._records: list[CompanyRecord] = []
        for record in records or []:
            self.add(record)

    def add(self, record: CompanyRecord) -> None:
        self._by_id[record.company_id] = record
        self._records.append(record)

    def get_by_id(self, company_id: str) -> CompanyRecord | None:
        return self._by_id.get(company_id)

    def get_by_dwh_id(self, dwh_id: str) -> CompanyRecord | None:
        for record in self._records:
            if str(record.raw.get("dwhId", "")) == str(dwh_id):
                return record
        return None

    def search_by_name(self, name: str) -> list[CompanyRecord]:
        needle = name.strip().lower()
        return [r for r in self._records if needle in r.company_name.lower()]

    def search_by_domain(self, domain: str) -> list[CompanyRecord]:
        needle = domain.strip().lower()
        return [r for r in self._records if needle in r.raw.get("companyWebsiteDomain", "").lower()]


class ElasticCompanyDataSource:
    """CompanyDataSource backed by real Elasticsearch. Configure via
    ES_HOST / ES_INDEX / ES_API_KEY. Auth uses a `Token` header (not the
    standard Authorization: ApiKey scheme - the gateway only accepts
    `Token`), with verify=False and a few retries for transient 404s."""

    _TIMEOUT_SECONDS = 30
    _MAX_ATTEMPTS = 3
    _RETRY_DELAY_SECONDS = 2

    def __init__(
        self,
        host: str | None = None,
        index: str | None = None,
        api_key: str | None = None,
    ):
        self.host = host or os.environ.get("ES_HOST")
        self.index = index or os.environ.get("ES_INDEX")
        self.api_key = api_key or os.environ.get("ES_API_KEY")

        if not self.host or not self.index or not self.api_key:
            raise RuntimeError(
                "ElasticCompanyDataSource: ES_HOST, ES_INDEX, and ES_API_KEY "
                "must all be set (env vars or constructor args)."
            )
        if not self.host.endswith("/"):
            self.host += "/"

    @staticmethod
    def _to_record(hit: dict) -> CompanyRecord:
        source = hit.get("_source", {})
        # .get(key, default) ignores `default` when the key exists but is null.
        company_id = source.get("companyId")
        if company_id is None:
            company_id = hit.get("_id", "")
        return CompanyRecord(
            company_id=str(company_id),
            company_name=source.get("companyName", ""),
            raw=source,
        )

    def _search(self, query: dict, size: int | None = None) -> list[dict]:
        body = {"query": query}
        if size is not None:
            body["size"] = size

        last_error = None
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(
                    f"{self.host}{self.index}/_search",
                    headers={"Token": self.api_key},
                    json=body,
                    verify=False,
                    timeout=self._TIMEOUT_SECONDS,
                )
            except requests.RequestException as exc:
                last_error = f"Could not reach Elasticsearch ({self.host}): {exc}"
            else:
                if resp.ok:
                    return (resp.json().get("hits") or {}).get("hits") or []
                last_error = f"Elasticsearch search failed (HTTP {resp.status_code}): {resp.text}"

            if attempt < self._MAX_ATTEMPTS:
                time.sleep(self._RETRY_DELAY_SECONDS)

        raise RuntimeError(f"{last_error} (gave up after {self._MAX_ATTEMPTS} attempts)")

    def get_by_id(self, company_id: str) -> CompanyRecord | None:
        hits = self._search({"term": {"companyId": company_id}}, size=1)
        return self._to_record(hits[0]) if hits else None

    def get_by_dwh_id(self, dwh_id: str) -> CompanyRecord | None:
        hits = self._search({"term": {"dwhId": dwh_id}}, size=1)
        return self._to_record(hits[0]) if hits else None

    def search_by_name(self, name: str) -> list[CompanyRecord]:
        hits = self._search({"match": {"companyName": name}})
        return [self._to_record(hit) for hit in hits]

    def search_by_domain(self, domain: str) -> list[CompanyRecord]:
        hits = self._search({"match": {"companyWebsiteDomain": domain}})
        return [self._to_record(hit) for hit in hits]
