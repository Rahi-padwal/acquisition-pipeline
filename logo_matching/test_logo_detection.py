"""Small-sample verification script for logo-based OCR acquisition detection,
sourcing logo images directly from crustdata's S3 URLs (logoS3Path in
Elasticsearch) rather than a local image folder.

Purpose: manually spot-check (a) PaddleOCR's text-extraction accuracy on
real, low-quality LinkedIn-scraped logo images, and (b) whether the keyword
patterns in detection.py correctly fire on what OCR actually returns -
BEFORE any decision to scale this to the full ~700K logo set. Matches the
"test small before scaling" approach already used for every other stage in
this project (see combined_matching/s3_detection.py, run_pipeline_on_file.py).

Does NOT write to combined_acquisitions or any database - print-only.

Pipeline per dwhId:
  1. Resolve a logo URL either directly from the input file (a
     "dwhId,logoS3Path" CSV row - see _read_input) or, for a plain
     dwhId-only row, by looking up companyName + logoS3Path in
     Elasticsearch (es_client.py). CSV rows skip the ES round-trip
     entirely since the URL is already on hand.
  2. If no logo URL is on record either way, skip - logged as "no_logo",
     no HTTP request made at all.
  3. Otherwise fetch the image over HTTP straight into memory
     (fetch_logo.py) - HTTP 200 -> decode and continue; HTTP 403 -> logged
     separately as "access_denied" (see fetch_logo.py's docstring: on this
     bucket 403 also covers plain-missing keys, not just real permission
     issues, which is exactly the ambiguity worth investigating later);
     anything else (404, 5xx, timeout, connection error, or an
     undecodable 200 body) -> "fetch_failed".
  4. On a decoded image, run OCR (ocr.py) - the image array is never
     written to disk.
  5. Only OCR lines at/above --min-confidence feed the keyword detector
     (detection.py, unchanged) - lower-confidence lines still get printed
     for spot-checking, just not trusted for matching.

Usage:
    python test_logo_detection.py [--input FILE] [--sample N] [--seed N] [--min-confidence F]

--input accepts two row formats, auto-detected per line:
  - "dwhId,logoS3Path"  - URL used directly, no ES lookup (companyName is
    unknown in this case, so the detector's self-reference guard is a
    no-op for these rows - see detection.py)
  - "dwhId"             - looked up in Elasticsearch for companyName + URL
Defaults to test_data.csv (dwhId,logoS3Path pairs) if present next to this
script, else ../dwhids_1987.txt (plain dwhId list, one per row, no header -
same convention as combined_matching/run_pipeline_on_file.py).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from detection import extract_acquirer_from_logo_text
from es_client import fetch_logo_lookups
from fetch_logo import fetch_logo_image
from ocr import DEFAULT_MIN_CONFIDENCE, run_ocr_array

# Windows consoles default to a codepage (e.g. cp1252) that can't encode
# every Unicode character OCR might return (CJK, accented Latin, etc. show
# up in real logo text) - reconfigure so a rare character doesn't crash an
# otherwise-successful run partway through.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = (
    _MODULE_DIR / "test_data.csv" if (_MODULE_DIR / "test_data.csv").is_file() else _MODULE_DIR.parent / "dwhids_1987.txt"
)


def _read_input(path: Path) -> dict[str, str | None]:
    """Returns {dwhId: logoS3Path_or_None}. A row with a comma is treated
    as "dwhId,logoS3Path" (URL known up front); a bare row is a dwhId to
    resolve via ES later."""
    rows: dict[str, str | None] = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            dwh_id = row[0].strip()
            url = row[1].strip() if len(row) > 1 and row[1].strip() else None
            rows[dwh_id] = url
    return rows


def main(input_path: Path, sample_size: int, seed: int, min_confidence: float) -> None:
    if not input_path.is_file():
        raise SystemExit(f"Not a file: {input_path}")

    all_rows = _read_input(input_path)
    print(f"Loaded {len(all_rows)} dwhId(s) from {input_path}")
    if not all_rows:
        return

    all_ids = list(all_rows.keys())
    rng = random.Random(seed)
    sample = all_ids if len(all_ids) <= sample_size else rng.sample(all_ids, sample_size)
    sample.sort()
    print(f"Sampling {len(sample)} dwhId(s) (seed={seed}), min OCR confidence={min_confidence}\n")

    # Only fall back to ES for rows that didn't already carry a URL.
    need_es_lookup = [dwh_id for dwh_id in sample if all_rows[dwh_id] is None]
    es_lookups = fetch_logo_lookups(need_es_lookup) if need_es_lookup else {}
    if need_es_lookup:
        print(f"Looked up logoS3Path in Elasticsearch for {len(need_es_lookup)} dwhId(s) with no URL in the input file\n")

    outcome_counts: dict[str, int] = {}
    matches: list[dict] = []

    for i, dwh_id in enumerate(sample, start=1):
        url_from_file = all_rows[dwh_id]
        if url_from_file is not None:
            logo_url, company_name, found_in_es = url_from_file, None, None
        else:
            lookup = es_lookups[dwh_id]
            logo_url, company_name, found_in_es = lookup.logo_s3_path, lookup.company_name, lookup.found_in_es

        label = f"[{i}/{len(sample)}] dwhId={dwh_id} companyName={company_name!r}"

        fetch_result = fetch_logo_image(logo_url)
        outcome_counts[fetch_result.outcome] = outcome_counts.get(fetch_result.outcome, 0) + 1

        if fetch_result.outcome == "no_logo":
            print(f"{label}\n    no_logo: {fetch_result.detail} (found_in_es={found_in_es})\n")
            continue
        if fetch_result.outcome == "access_denied":
            print(f"{label}\n    access_denied: {fetch_result.detail} url={logo_url!r}\n")
            continue
        if fetch_result.outcome == "fetch_failed":
            print(f"{label}\n    fetch_failed: {fetch_result.detail} url={logo_url!r}\n")
            continue

        # outcome == "ok"
        print(f"{label}\n    url={logo_url}")
        ocr_result = run_ocr_array(fetch_result.image)
        if not ocr_result.lines:
            print("    OCR text: (none detected)")
        else:
            print(f"    OCR text (all lines): {ocr_result.text!r}")
            for text, score in ocr_result.lines:
                trusted = "trusted" if score >= min_confidence else "below threshold"
                print(f"      line={text!r} score={score:.2f} ({trusted})")

        high_conf_text = ocr_result.high_confidence_text(min_confidence)
        detection = extract_acquirer_from_logo_text(high_conf_text, company_name)
        if detection is not None:
            print(
                f"    MATCH: pattern={detection['matchedPattern']} "
                f"acquirer={detection['acquirerCompany']!r} "
                f"confidence={detection['confidenceScore']} "
                f"rawText={detection['rawText']!r}"
            )
            matches.append({"dwhId": dwh_id, "companyName": company_name, **detection})
        print()

    print("--- Summary ---")
    print(f"  dwhIds sampled       = {len(sample)}")
    for label in ("ok", "no_logo", "access_denied", "fetch_failed"):
        print(f"  {label:<15} = {outcome_counts.get(label, 0)}")
    print(f"  matches found        = {len(matches)}")
    if matches:
        print("\nMatched companies:")
        for m in matches:
            print(
                f"  dwhId={m['dwhId']} companyName={m['companyName']!r}: pattern={m['matchedPattern']} "
                f"acquirer={m['acquirerCompany']!r} confidence={m['confidenceScore']}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=_DEFAULT_INPUT, help="File of 'dwhId,logoS3Path' or plain dwhId rows, no header"
    )
    parser.add_argument("--sample", type=int, default=20, help="Number of dwhIds to sample (default 20)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling (default 0)")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum per-line OCR confidence trusted for keyword matching (default {DEFAULT_MIN_CONFIDENCE})",
    )
    args = parser.parse_args()
    main(args.input, args.sample, args.seed, args.min_confidence)
