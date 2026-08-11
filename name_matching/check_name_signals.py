"""Detect acquisitions from a company's NAME alone, via a narrow regex set."""

from __future__ import annotations
import re
from data_source import CompanyDataSource, CompanyRecord

# Capital letter + up to 50 more name-shaped chars.
_NAME_CAPTURE = r"[A-Z][\w&.,' -]{1,50}?"

# (name, pattern). Anchored end-to-end - the whole field is matched, not a
# sentence in a document. (?i:...) scopes case-insensitivity to keywords only,
# so the acquirer's [A-Z] anchor still requires a real capital letter.
_NAME_ACQUIRER_PATTERNS = [
    (
        "subsidiary_of",
        re.compile(
            # [\s-]+ so "wholly-owned" (hyphenated) matches too.
            r"^(?P<self>.+?),?\s+(?i:(?:is\s+)?an?\s+(?:wholly[\s-]+owned[\s-]+)?subsidiary\s+of)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "owned_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:is\s+)?owned\s+by)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "part_of",
        re.compile(
            r"^(?P<self>.+?)[,\-–—]?\s*(?i:part\s+of)\s+(?i:the\s+)?"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")(?:\s+(?i:group))?\)?\s*$"
        ),
    ),
    (
        "a_x_company",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+(?i:company)\)?\s*$"
        ),
    ),
    (
        # Reverse word order from "a_x_company": "X, a company of Y".
        "a_company_of",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?\s+company\s+of)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "a_x_brand",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+(?i:brand)\)?\s*$"
        ),
    ),
    (
        "joined_forces_with",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:has\s+)?joined\s+forces\s+with)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "acquired_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:(?:has\s+been|was|is)\s+)?acquired\s+by)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "powered_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:powered\s+by)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "operating_unit_of",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:an?\s+operating\s+unit\s+of)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "division_of",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:an?\s+division\s+of)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        # Reversed group roles: acquirer comes BEFORE "trading as", self after.
        "trading_as",
        re.compile(
            r"^(?P<acquirer>" + _NAME_CAPTURE + r"),?\s+(?i:trading\s+as)\s+(?P<self>.+?)\)?\s*$"
        ),
    ),
    (
        # Reversed roles again: "Google acquires Nest" - acquirer stated first.
        "acquirer_acquires_self",
        re.compile(
            r"^(?P<acquirer>" + _NAME_CAPTURE + r"),?\s+(?i:acquires)\s+(?P<self>.+?)\)?\s*$"
        ),
    ),
    (
        # "Nest by Google" - riskier, "by" is common outside acquisitions too.
        "self_by_acquirer",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:by)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
]

# Unanchored - scans a parenthetical fragment inside a larger string, e.g.
# "Series A - Advizzo (part of Calisen Group)".
_PARENTHETICAL_PART_OF = re.compile(r"\((?i:part\s+of\s+(?:the\s+)?)(?P<acquirer>" + _NAME_CAPTURE + r")\)")

# Generic filler a match can capture instead of a real company name.
_GENERIC_TERM_DENYLIST = {
    "saas",
    "software",
    "technology",
    "tech",
    "cloud",
    "digital",
    "platform",
    "solution",
    "solutions",
    "service",
    "services",
    "leading",
    "global",
    "enterprise",
    "startup",
    "company",
    "business",
    "product",
    "brand",
    "group",
}


def _names_match(a: str, b: str) -> bool:
    """Case-insensitive equality, treating one name containing the other as a match."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


def _is_valid_acquirer_candidate(candidate: str, self_name: str) -> bool:
    candidate = candidate.strip()
    if not (3 <= len(candidate) <= 50):
        return False
    if not candidate[0].isupper():
        return False
    if candidate.lower() in _GENERIC_TERM_DENYLIST:
        return False
    if _names_match(candidate, self_name):
        return False
    return True


def find_acquirer_in_name(company_name: str) -> dict | None:
    """Return {candidate_name, matched_pattern, self_name, raw_name} for the
    first valid match, or None."""

    name = (company_name or "").strip()
    if not name:
        return None

    for pattern_name, pattern in _NAME_ACQUIRER_PATTERNS:
        match = pattern.match(name)
        if not match:
            continue

        acquirer = match.group("acquirer").strip().rstrip(").,")
        self_name = match.group("self").strip()
        if not _is_valid_acquirer_candidate(acquirer, self_name):
            continue

        return {
            "candidate_name": acquirer,
            "matched_pattern": pattern_name,
            "self_name": self_name,
            "raw_name": name,
        }

    return None


def _check_structured_fields(raw: dict, record_company_id: str) -> dict | None:
    """Prefer Elastic's own isAcquired / acquirerCompanyName fields over
    regex-parsing the name, when present.

    Self-reference is guarded via acquirerCompanyId vs the record's own
    companyId, not a name substring check - companyName routinely already
    contains the acquirer's name (e.g. "Authomize, now part of Delinea").
    """
    if not raw.get("isAcquired"):
        return None

    acquired_by = (raw.get("fundingAndInvestment") or {}).get("acquiredBy") or {}
    # Nested fields can come back as a list even for a single value.
    if isinstance(acquired_by, list):
        acquired_by = acquired_by[0] if acquired_by else {}

    acquirer = (acquired_by.get("acquirerCompanyName") or "").strip()
    acquirer_id = str(acquired_by.get("acquirerCompanyId") or "")
    if not acquirer:
        return None
    if acquirer_id and acquirer_id == str(record_company_id):
        return None

    company_name = raw.get("companyName", "")
    return {
        "candidate_name": acquirer,
        "matched_pattern": "structured_field",
        "self_name": company_name,
        "raw_name": company_name,
    }


def _check_funding_milestones(raw: dict, self_name: str) -> dict | None:
    """Fallback for stale/empty isAcquired+acquiredBy: scans funding-round
    names for a "(part of X)" parenthetical, e.g. "Series A - Advizzo (part
    of Calisen Group)"."""
    funding = raw.get("fundingAndInvestment") or {}
    if isinstance(funding, list):
        funding = funding[0] if funding else {}

    milestones = funding.get("fundingMilestonesTimeseries") or []
    if isinstance(milestones, dict):
        milestones = [milestones]

    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        funding_round = milestone.get("fundingRound")
        texts = funding_round if isinstance(funding_round, list) else [funding_round]

        for text in texts:
            match = _PARENTHETICAL_PART_OF.search(text or "")
            if not match:
                continue
            acquirer = match.group("acquirer").strip().rstrip(").,")
            if not _is_valid_acquirer_candidate(acquirer, self_name):
                continue
            return {
                "candidate_name": acquirer,
                "matched_pattern": "funding_round_parenthetical",
                "self_name": self_name,
                "raw_name": text,
            }

    return None


def check_company_record(record) -> dict | None:
    """Check a CompanyRecord for an acquirer: structured fields, then name
    regex, then funding-round parenthetical, in that order."""
    structured = _check_structured_fields(record.raw, record.company_id)
    if structured is not None:
        return structured

    name_match = find_acquirer_in_name(record.company_name)
    if name_match is not None:
        return name_match

    return _check_funding_milestones(record.raw, record.company_name)


def _domain_key(record: CompanyRecord) -> str:
    return record.raw.get("companyWebsiteDomain", "")


def _search_and_check(
    query: str,
    records: list[CompanyRecord],
    key=lambda r: r.company_name,
) -> dict:
    """A name/domain search can return several different companies (e.g. many
    sub-brands sharing one parent domain). Check EVERY candidate and return
    the first with real evidence, trying an exact match on `key` first, then
    result order. Falls back to "not acquired" only if none have evidence.
    """
    query_lower = query.strip().lower()
    exact = [r for r in records if key(r).strip().lower() == query_lower]
    ordered = exact + [r for r in records if r not in exact]

    for record in ordered:
        match = check_company_record(record)
        if match is not None:
            return _positive_result(record, match)

    return _not_acquired_result(ordered[0])


def _positive_result(record: CompanyRecord, match: dict) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": match["candidate_name"],
        "evidence": [
            {
                "candidate_name": match["candidate_name"],
                "matched_pattern": match["matched_pattern"],
                "raw_name": match["raw_name"],
            }
        ],
        "status": "checked",
    }


def _not_found_result(query: str) -> dict:
    return {
        "company_id": None,
        "company_name": query,
        "is_acquired": False,
        "likely_acquirer": None,
        "evidence": [],
        "status": "not_found",
    }


def _not_acquired_result(record: CompanyRecord) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": False,
        "likely_acquirer": None,
        "evidence": [],
        "status": "checked",
    }


def run_name_search_check(company_name: str, data_source: CompanyDataSource) -> dict:
    """Search `data_source` for `company_name` and check the best match for
    an acquirer. See _search_and_check() for how multi-hit results are
    resolved; status is "checked" or "not_found"."""
    results = data_source.search_by_name(company_name)
    if not results:
        return _not_found_result(company_name)

    return _search_and_check(company_name, results)


def run_domain_search_check(company_domain: str, data_source: CompanyDataSource) -> dict:
    """Same as run_name_search_check(), keyed by companyWebsiteDomain. ES's
    `match` query is analyzed, not exact, so results are filtered down to
    ones whose domain genuinely contains the query first."""
    results = data_source.search_by_domain(company_domain)
    query_lower = company_domain.strip().lower()
    results = [r for r in results if query_lower in _domain_key(r).lower()]
    if not results:
        return _not_found_result(company_domain)

    return _search_and_check(company_domain, results, key=_domain_key)


def run_dwh_id_check(dwh_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `dwh_id` (a unique key) and check it for an acquirer. No
    disambiguation needed since dwhId identifies exactly one record."""
    record = data_source.get_by_dwh_id(dwh_id)
    if record is None:
        return _not_found_result(dwh_id)

    match = check_company_record(record)
    if match is None:
        return _not_acquired_result(record)

    return _positive_result(record, match)


def run_name_check(company_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `company_id` and check its companyName for an embedded acquirer."""
    record = data_source.get_by_id(company_id)
    if record is None:
        return {
            "company_id": company_id,
            "company_name": None,
            "is_acquired": False,
            "likely_acquirer": None,
            "evidence": [],
            "status": "not_found",
        }

    match = find_acquirer_in_name(record.company_name)
    if match is None:
        return {
            "company_id": company_id,
            "company_name": record.company_name,
            "is_acquired": False,
            "likely_acquirer": None,
            "evidence": [],
            "status": "checked",
        }

    return {
        "company_id": company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": match["candidate_name"],
        "evidence": [
            {
                "candidate_name": match["candidate_name"],
                "matched_pattern": match["matched_pattern"],
                "raw_name": match["raw_name"],
            }
        ],
        "status": "checked",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python check_name_signals.py '<company name string>'")
        sys.exit(1)

    result = find_acquirer_in_name(sys.argv[1])
    print(result if result else "No acquirer detected in that name.")
