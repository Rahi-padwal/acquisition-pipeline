"""Run the combined acquisition-detection pipeline (companyName-based +
description-based) over a list of dwhIds read from a CSV/Excel file, and
write the results to an Excel file.

Reuses the exact same functions run_full_batch.py uses - no detection logic
duplicated here:
- check_name_signals.check_company_record() (name-based: structured ES
  fields, companyName regex, funding-round parentheticals)
- description_detection_api.extract_acquirer_from_description() (description
  phrase patterns)
- test_combined_pipeline._combine_results() (merges both into one verdict,
  including the review-bucket floor and trust-tiered conflict handling)

Input file: one dwhId per row in the first column, no header expected (a
plain list of dwhIds, like the CSV/Excel files used earlier this session).

Usage:
    python run_pipeline_on_file.py "input.csv" "output.xlsx"
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import check_company_record  # noqa: E402 (from name_matching)
from data_source import CompanyDataSource, ElasticCompanyDataSource  # noqa: E402 (from name_matching)
from description_detection_api import extract_acquirer_from_description  # noqa: E402 (from description_matching)

from test_combined_pipeline import _combine_results  # noqa: E402

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

MAX_WORKERS = 10

OUTPUT_COLUMNS = [
    "dwhId", "companyName", "isAcquired", "acquirerCompany", "source",
    "matchedPattern", "confidenceScore", "reviewStatus",
]


def _read_dwh_ids(path: str) -> list[str]:
    read = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = read(path, header=None, dtype=str)
    return [v.strip() for v in df.iloc[:, 0].dropna().tolist() if v.strip()]


def process_dwh_id(dwh_id: str, data_source: CompanyDataSource) -> dict[str, Any]:
    """Never raises - failures come back as isAcquired=None instead of dropping."""
    try:
        record = data_source.get_by_dwh_id(dwh_id)
        if record is None:
            return {
                "dwhId": dwh_id, "companyName": None, "isAcquired": None,
                "acquirerCompany": None, "source": None, "matchedPattern": None,
                "confidenceScore": None, "reviewStatus": None,
            }
        name_match = check_company_record(record)
        description_match = extract_acquirer_from_description(record.raw.get("description"))
        combined = _combine_results(name_match, description_match, record.raw)
        return {"dwhId": dwh_id, "companyName": record.company_name, **combined}
    except Exception as exc:
        print(f"  dwhId {dwh_id}: failed to process - {exc}")
        return {
            "dwhId": dwh_id, "companyName": None, "isAcquired": None,
            "acquirerCompany": None, "source": None, "matchedPattern": None,
            "confidenceScore": None, "reviewStatus": None,
        }


def main(input_path: str, output_path: str) -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_dwh_id, dwh_id, data_source): dwh_id for dwh_id in dwh_ids}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  processed {done}/{len(dwh_ids)}")

    order = {dwh_id: i for i, dwh_id in enumerate(dwh_ids)}
    results.sort(key=lambda r: order.get(r["dwhId"], len(dwh_ids)))

    df_out = pd.DataFrame(results, columns=OUTPUT_COLUMNS)
    df_out.to_excel(output_path, index=False)

    acquired = (df_out["isAcquired"] == 1).sum()
    likely_acquired = (df_out["reviewStatus"] == "likely_acquired").sum()
    not_found = df_out["isAcquired"].isna().sum()
    print(f"Wrote {len(df_out)} rows to {output_path}")
    print(f"  acquired={acquired} likely_acquired={likely_acquired} not_acquired={len(df_out) - acquired - likely_acquired - not_found} not_found/errors={not_found}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python run_pipeline_on_file.py "input.csv" "output.xlsx"')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
