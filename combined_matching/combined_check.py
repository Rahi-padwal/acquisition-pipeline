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

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

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


# --- Post-hoc structured-field confidence boost ------------------------------
#
# Manual spot-checking found that when a row's confidence is low/moderate
# because the text-based pattern match is weak or absent, the raw ES field
# fundingAndInvestment.acquiredBy.acquirerCompanyName is often still correct
# even with ZERO corroborating acquisition language anywhere in the company's
# own text (e.g. Nuvance Health, acquired by Northwell Health - no
# acquisition language in the description at all, yet the structured field
# was accurate). This step nudges the score up in that situation. It is a
# blunt, manually-tuned correction (not a new detector), which is exactly why
# the thresholds/targets are pulled out as named constants here instead of
# being buried as magic numbers inline - keep it that way so it's easy to
# find and retune later.
#
# Runs strictly AFTER the base pattern score and all existing modifiers
# (corroboration bonus, structured-field-uncorroborated cap, substring-
# rejection, conflict cap) have already produced `score`/`notes` - this is an
# additional final-adjustment layer, not a replacement for any of that.
_STRUCTURED_FIELD_BOOST_LOW_MAX = 0.5   # score <= this -> boosted to _LOW_TARGET
_STRUCTURED_FIELD_BOOST_MID_MAX = 0.7   # _LOW_MAX < score <= this -> boosted to _MID_TARGET
_STRUCTURED_FIELD_BOOST_LOW_TARGET = 0.7
_STRUCTURED_FIELD_BOOST_MID_TARGET = 0.8

# Set by check_name_signals._score_match() when a structured_field match's
# acquirer name doesn't appear anywhere in the company's own text - see that
# module. Duplicated here as a literal (rather than imported) since it's a
# string constant, not shared logic; keep the two in sync if it ever changes.
_STRUCTURED_FIELD_UNCORROBORATED_NOTE = "structured_field_uncorroborated"


def _structured_field_acquirer_name(raw: dict) -> str:
    """Read fundingAndInvestment.acquiredBy.acquirerCompanyName straight off
    the raw record, same lookup check_name_signals._check_structured_fields()
    does. Independent of whatever ended up in acquirerCompany/likely_acquirer
    - this function only asks "is the structured field itself populated?"."""
    acquired_by = (raw.get("fundingAndInvestment") or {}).get("acquiredBy") or {}
    if isinstance(acquired_by, list):
        acquired_by = acquired_by[0] if acquired_by else {}
    return (acquired_by.get("acquirerCompanyName") or "").strip()


def _apply_structured_field_confidence_boost(
    score: float | None, notes: str | None, raw: dict
) -> tuple[float | None, str | None]:
    """Final post-processing adjustment: bump `score` up when the raw
    structured field has an acquirer name, per the two thresholds above.
    Must be called LAST, after every other score/notes computation for this
    row is finished.

    Only adjusts the score - never touches acquirerCompany, source,
    matchedPattern, or isAcquired. Whether to also backfill acquirerCompany
    from the structured field when it's currently empty is a separate
    question, out of scope here.

    Deliberately a no-op when `notes` already says the structured-field-
    uncorroborated cap fired (see check_name_signals._score_match): that cap
    and this boost are two different conclusions about the exact same
    signal (structured field present, no text corroboration) pulling in
    opposite directions, and nothing has specified which one should win. So
    rather than silently guessing, this logs the conflict and leaves the
    existing (capped) score untouched - revisit explicitly if that turns out
    to happen a lot.
    """
    if score is None:
        return score, notes

    structured_acquirer = _structured_field_acquirer_name(raw)
    if not structured_acquirer:
        return score, notes

    if notes == _STRUCTURED_FIELD_UNCORROBORATED_NOTE:
        logger.warning(
            "structured-field confidence boost skipped: row already capped by "
            "structured_field_uncorroborated (score=%s, structured acquirer=%r) "
            "- priority between the cap and the boost is undecided, cap wins",
            score,
            structured_acquirer,
        )
        return score, notes

    if score <= _STRUCTURED_FIELD_BOOST_LOW_MAX:
        return _STRUCTURED_FIELD_BOOST_LOW_TARGET, notes
    if score <= _STRUCTURED_FIELD_BOOST_MID_MAX:
        return _STRUCTURED_FIELD_BOOST_MID_TARGET, notes
    return score, notes


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

        # Final post-processing step, applied after every score/notes
        # computation above - see _apply_structured_field_confidence_boost()'s
        # docstring. Skipped when the match ITSELF is structured_field: in
        # that case "is the structured field populated?" is tautologically
        # always true (that's literally the data that produced this match),
        # so calling the boost here is circular and does nothing but log a
        # meaningless "conflict" warning on every single uncorroborated
        # structured_field row in the dataset. The boost is only meaningful
        # as EXTRA corroboration for a match that came from a different
        # source (powered_by, joined_forces, description) and happens to
        # also have a populated structured field.
        if name_match["matched_pattern"] != "structured_field":
            score, notes = _apply_structured_field_confidence_boost(score, notes, record.raw)

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
        description_score, description_notes = _apply_structured_field_confidence_boost(
            description_match["confidenceScore"], None, record.raw
        )
        return _positive_result(
            record,
            description_acquirer,
            description_match["matchedPattern"],
            description_match["rawText"],
            source="description",
            confidence_score=description_score,
            confidence_notes=description_notes,
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
