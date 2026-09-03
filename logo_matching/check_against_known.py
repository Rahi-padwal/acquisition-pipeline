"""Measure logo-OCR detection against companies stages 1-3 ALREADY resolved.

This is the "check it first" step before wiring logo detection into the
main pipeline: rather than guessing at accuracy, run the real ES -> S3 ->
OCR -> detect pipeline (unchanged from test_logo_detection.py's modules)
against dwhIds where combined_matching's existing pipeline already found a
confirmed acquirer, and see whether the logo signal agrees, disagrees, or
says nothing.

Read-only against combined_matching/test_combined_pipeline.db - this
script only SELECTs from it, never writes, and lives entirely in
logo_matching/ (no other project file is modified).

Three outcomes per known-positive company:
  - "corroborated" - logo detector found an acquirer, and it matches the
    already-known one (loose name comparison, see _acquirer_names_match)
  - "conflict"      - logo detector found a DIFFERENT acquirer
  - "no_signal"      - logo detector found nothing (most logos simply
    don't name their acquirer - not a failure, just no evidence either way)
Fetch-side failures (no_logo/access_denied/fetch_failed) are counted
separately, same as test_logo_detection.py.

Usage:
    python check_against_known.py [--db PATH]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

from detection import extract_acquirer_from_logo_text
from es_client import fetch_logo_lookups
from fetch_logo import fetch_logo_image
from ocr import run_ocr_array

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_DEFAULT_DB = Path(__file__).resolve().parent.parent / "combined_matching" / "test_combined_pipeline.db"

_LEGAL_SUFFIX_RE = re.compile(r"\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings|gmbh|plc)\b\.?", re.I)


def _normalize(name: str) -> str:
    name = _LEGAL_SUFFIX_RE.sub(" ", name or "")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _acquirer_names_match(a: str, b: str) -> bool:
    """Loose match: same convention used throughout this project
    (name_matching/check_name_signals.py's _company_identity_match,
    s3_detection.py's _is_self_reference) - strip legal suffixes/
    punctuation, then accept if either normalized name contains the other."""
    na, nb = _normalize(a), _normalize(b)
    if len(na) < 2 or len(nb) < 2:
        return False
    return na in nb or nb in na


def _load_known_positives(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT dwhId, companyName, acquirerCompany, source, matchedPattern, confidenceScore "
        "FROM combined_acquisitions WHERE acquirerCompany IS NOT NULL AND acquirerCompany != ''"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main(db_path: Path) -> None:
    known = _load_known_positives(db_path)
    print(f"Loaded {len(known)} known-positive dwhId(s) from {db_path} (stages 1-3 already found an acquirer)\n")
    if not known:
        return

    dwh_ids = [k["dwhId"] for k in known]
    lookups = fetch_logo_lookups(dwh_ids)

    outcome_counts: dict[str, int] = {}
    comparison_counts: dict[str, int] = {"corroborated": 0, "conflict": 0, "no_signal": 0}
    conflicts: list[dict] = []
    corroborations: list[dict] = []

    for i, k in enumerate(known, start=1):
        dwh_id = k["dwhId"]
        lookup = lookups[dwh_id]
        label = (
            f"[{i}/{len(known)}] dwhId={dwh_id} companyName={k['companyName']!r} "
            f"knownAcquirer={k['acquirerCompany']!r} (source={k['source']}, pattern={k['matchedPattern']}, "
            f"confidence={k['confidenceScore']})"
        )
        print(label)

        fetch_result = fetch_logo_image(lookup.logo_s3_path)
        outcome_counts[fetch_result.outcome] = outcome_counts.get(fetch_result.outcome, 0) + 1

        if fetch_result.outcome != "ok":
            print(f"    {fetch_result.outcome}: {fetch_result.detail} url={lookup.logo_s3_path!r}\n")
            continue

        ocr_result = run_ocr_array(fetch_result.image)
        print(f"    OCR text: {ocr_result.text!r}")
        high_conf_text = ocr_result.high_confidence_text()
        detection = extract_acquirer_from_logo_text(high_conf_text, k["companyName"])

        if detection is None:
            comparison_counts["no_signal"] += 1
            print("    logo detector: no match (no_signal)\n")
            continue

        logo_acquirer = detection["acquirerCompany"]
        if _acquirer_names_match(logo_acquirer, k["acquirerCompany"]):
            comparison_counts["corroborated"] += 1
            corroborations.append({**k, "logoAcquirer": logo_acquirer, "logoConfidence": detection["confidenceScore"]})
            print(
                f"    logo detector: CORROBORATED - found {logo_acquirer!r} "
                f"(pattern={detection['matchedPattern']}, confidence={detection['confidenceScore']}) "
                f"matches known acquirer {k['acquirerCompany']!r}\n"
            )
        else:
            comparison_counts["conflict"] += 1
            conflicts.append({**k, "logoAcquirer": logo_acquirer, "logoConfidence": detection["confidenceScore"]})
            print(
                f"    logo detector: CONFLICT - found {logo_acquirer!r} "
                f"(pattern={detection['matchedPattern']}, confidence={detection['confidenceScore']}) "
                f"but known acquirer is {k['acquirerCompany']!r}\n"
            )

    print("--- Fetch outcomes ---")
    for label in ("ok", "no_logo", "access_denied", "fetch_failed"):
        print(f"  {label:<15} = {outcome_counts.get(label, 0)}")

    print("\n--- Comparison against known answers (of successfully OCR'd images) ---")
    for label in ("corroborated", "conflict", "no_signal"):
        print(f"  {label:<15} = {comparison_counts[label]}")

    if corroborations:
        print("\nCorroborated:")
        for c in corroborations:
            print(f"  {c['companyName']!r}: known={c['acquirerCompany']!r} logo={c['logoAcquirer']!r}")
    if conflicts:
        print("\nConflicts (worth manually reviewing):")
        for c in conflicts:
            print(f"  {c['companyName']!r}: known={c['acquirerCompany']!r} logo={c['logoAcquirer']!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB, help="Path to combined_matching's SQLite DB (read-only)")
    args = parser.parse_args()
    main(args.db)
