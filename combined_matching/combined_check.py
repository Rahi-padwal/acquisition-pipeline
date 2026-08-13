"""Combined acquisition-detection pipeline: dwhId in, one check out.

Runs the name_matching pipeline first (check_name_signals.check_company_record
- structured ES fields, then companyName regex, then funding-round
parentheticals). Only when that comes back empty (is_acquired would be
False) does it fall back to description_matching's phrase-pattern detection
over the same record's `description` field. One ES fetch total - both
checks run against the CompanyRecord already pulled by dwhId.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import check_company_record  # noqa: E402 (from name_matching)
from data_source import CompanyDataSource, CompanyRecord, ElasticCompanyDataSource  # noqa: E402 (from name_matching)
from description_detection_api import extract_acquirer_from_description  # noqa: E402 (from description_matching)


def _not_found_result(dwh_id: str) -> dict:
    return {
        "company_id": None,
        "company_name": dwh_id,
        "is_acquired": False,
        "likely_acquirer": None,
        "source": None,
        "evidence": [],
    }


def _not_acquired_result(record: CompanyRecord) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": False,
        "likely_acquirer": None,
        "source": None,
        "evidence": [],
    }


def _positive_result(
    record: CompanyRecord, candidate_name: str | None, matched_pattern: str, raw_name: str, source: str
) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": candidate_name,
        "source": source,
        "evidence": [
            {
                "candidate_name": candidate_name,
                "matched_pattern": matched_pattern,
                "raw_name": raw_name,
            }
        ],
    }


def run_combined_check(dwh_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `dwh_id` and check it for an acquirer: name-based signals
    first, description-based phrase patterns only if those find nothing."""
    record = data_source.get_by_dwh_id(dwh_id)
    if record is None:
        return _not_found_result(dwh_id)

    name_match = check_company_record(record)
    if name_match is not None:
        return _positive_result(
            record,
            name_match["candidate_name"],
            name_match["matched_pattern"],
            name_match["raw_name"],
            source="company_name",
        )

    description_match = extract_acquirer_from_description(record.raw.get("description"))
    if description_match is not None:
        return _positive_result(
            record,
            description_match["acquirerCompany"],
            description_match["matchedPattern"],
            description_match["rawText"],
            source="description",
        )

    return _not_acquired_result(record)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) != 2:
        print("Usage: python combined_check.py <dwhId>")
        sys.exit(1)

    data_source = ElasticCompanyDataSource(
        host=os.environ.get("ES_HOST"),
        index=os.environ.get("ES_INDEX"),
        api_key=os.environ.get("ES_API_KEY"),
    )
    print(run_combined_check(sys.argv[1], data_source))
