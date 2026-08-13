"""Combined acquisition-detection pipeline: dwhId in, one check out.

Runs the name_matching pipeline (check_name_signals.check_company_record -
structured ES fields, then companyName regex, then funding-round
parentheticals) AND description_matching's phrase-pattern detection over the
same record's `description` field, every time - see the note on
run_combined_check() for why both now always run instead of short-circuiting.
One ES fetch total - both checks run against the CompanyRecord already
pulled by dwhId.

Base confidence scores per matched_pattern live in each sub-module
(name_matching/check_name_signals.py's _PATTERN_BASE_CONFIDENCE,
description_matching/description_detection_api.py's own copy) since each
only has visibility into its own detector. This module is the one place
that sees BOTH detectors' results for the same record, so it's the one
place the cross-source modifiers (corroboration/conflict) can be applied -
see _apply_cross_source_modifiers() below.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import _company_identity_match, check_company_record  # noqa: E402 (from name_matching)
from data_source import CompanyDataSource, CompanyRecord, ElasticCompanyDataSource  # noqa: E402 (from name_matching)
from description_detection_api import extract_acquirer_from_description  # noqa: E402 (from description_matching)

# check_company_record() actually runs three distinct detectors in sequence
# (see its docstring) and tags each match with which one fired via
# matched_pattern. "source" must reflect that same distinction, not blanket-
# label everything it returns as "company_name" - a structured_field match
# came from ES's own isAcquired/acquiredBy fields, never from parsing the
# companyName string, and mislabeling it as "company_name" is misleading
# even when the extracted acquirer itself is correct.
_NAME_MATCHING_SOURCE_BY_PATTERN = {
    "structured_field": "structured_field",
    "funding_round_parenthetical": "funding_milestone",
}


def _source_for_name_match(matched_pattern: str) -> str:
    # Any other matched_pattern is one of the companyName regex names
    # (subsidiary_of, acquired_by, part_of, ...) - genuinely company_name-derived.
    return _NAME_MATCHING_SOURCE_BY_PATTERN.get(matched_pattern, "company_name")


# --- Cross-source confidence modifiers ---------------------------------------
#
# Only meaningful when BOTH detectors produced an actual candidate for the
# same record - this module is the only place that's ever known, since it's
# the only one calling both check_company_record() (name-based, including
# structured fields) and extract_acquirer_from_description() (description-
# based) on the same record. Per-detector base scores (and, for
# structured_field, the single-record "does the acquirer appear in this
# company's own text" modifier) are already baked into the score each
# sub-module hands back before this ever runs.
_CORROBORATION_BONUS = 0.05
_CORROBORATION_CAP = 0.98
_CONFLICT_CAP = 0.4


def _apply_cross_source_modifiers(
    primary_acquirer: str, primary_score: float, other_acquirer: str | None
) -> tuple[float, str | None]:
    """Adjust `primary_score` given the OTHER source's candidate (or None if
    it didn't find one). `primary_acquirer`/`primary_score` belong to
    whichever source wins as likely_acquirer (name-based takes priority when
    both fire - unchanged from before this scoring work). Returns
    (adjusted_score, confidence_notes)."""
    if other_acquirer is None:
        return primary_score, None
    if _company_identity_match(primary_acquirer, other_acquirer):
        return min(_CORROBORATION_CAP, primary_score + _CORROBORATION_BONUS), "corroborated_by_both_sources"
    return min(primary_score, _CONFLICT_CAP), "source_conflict"


def _not_found_result(dwh_id: str) -> dict:
    return {
        "company_id": None,
        "company_name": dwh_id,
        "is_acquired": False,
        "likely_acquirer": None,
        "source": None,
        "confidence_score": None,
        "confidence_notes": None,
        "evidence": [],
    }


def _not_acquired_result(record: CompanyRecord) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": False,
        "likely_acquirer": None,
        "source": None,
        "confidence_score": None,
        "confidence_notes": None,
        "evidence": [],
    }


def _positive_result(
    record: CompanyRecord,
    candidate_name: str | None,
    matched_pattern: str,
    raw_name: str,
    source: str,
    confidence_score: float,
    confidence_notes: str | None,
    extra_evidence: list[dict] | None = None,
) -> dict:
    evidence = [
        {
            "candidate_name": candidate_name,
            "matched_pattern": matched_pattern,
            "raw_name": raw_name,
        }
    ]
    if extra_evidence:
        evidence.extend(extra_evidence)

    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": candidate_name,
        "source": source,
        "confidence_score": confidence_score,
        "confidence_notes": confidence_notes,
        "evidence": evidence,
    }


def run_combined_check(dwh_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `dwh_id` and check it for an acquirer via both detectors.

    Both check_company_record() (name-based) and
    extract_acquirer_from_description() (description-based) always run now,
    even though only one of them ends up chosen as likely_acquirer (name-
    based still wins when both fire - that priority is unchanged). This is a
    deliberate change from the earlier short-circuit ("only check
    description if name-based found nothing"): the corroboration/conflict
    modifiers below are structurally unreachable unless both sources are
    evaluated on every record, so scoring them meaningfully requires it.
    """
    record = data_source.get_by_dwh_id(dwh_id)
    if record is None:
        return _not_found_result(dwh_id)

    name_match = check_company_record(record)
    description_match = extract_acquirer_from_description(record.raw.get("description"))
    # A match with no acquirerCompany (demoted "formerly", or the
    # consolidator heuristic in extract_acquirer_from_description) is
    # evidence of a name change, not of the subject being acquired - not a
    # real candidate for either a positive result or cross-source scoring.
    description_acquirer = (
        description_match["acquirerCompany"]
        if description_match is not None and description_match["acquirerCompany"] is not None
        else None
    )

    if name_match is not None:
        score, notes = _apply_cross_source_modifiers(
            name_match["candidate_name"], name_match["confidence_score"], description_acquirer
        )
        # No cross-source signal either way - fall back to whatever note
        # (if any) the name-matching side's own modifier already set, e.g.
        # "structured_field_uncorroborated".
        if notes is None:
            notes = name_match.get("confidence_notes")

        extra_evidence = None
        if notes == "source_conflict":
            extra_evidence = [
                {
                    "candidate_name": description_acquirer,
                    "matched_pattern": description_match["matchedPattern"],
                    "raw_name": description_match["rawText"],
                }
            ]

        return _positive_result(
            record,
            name_match["candidate_name"],
            name_match["matched_pattern"],
            name_match["raw_name"],
            source=_source_for_name_match(name_match["matched_pattern"]),
            confidence_score=score,
            confidence_notes=notes,
            extra_evidence=extra_evidence,
        )

    if description_acquirer is not None:
        # name_match is None here, so there's no second candidate to
        # corroborate or conflict with - cross-source modifiers can't apply.
        return _positive_result(
            record,
            description_acquirer,
            description_match["matchedPattern"],
            description_match["rawText"],
            source="description",
            confidence_score=description_match["confidenceScore"],
            confidence_notes=None,
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
