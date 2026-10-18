"""Run the FULL 3-stage acquisition-detection pipeline over a list of dwhIds
read from a CSV/Excel file: companyName -> description -> S3 web scrape, in
that order, writing results into the shared combined_acquisitions SQLite
table and an Excel file.

Stage 1+2 (companyName + description, ES-sourced) reuses the exact same
functions run_full_batch.py uses - check_name_signals.check_company_record(),
description_detection_api.extract_acquirer_from_description(), and
test_combined_pipeline._combine_results() - then writes that verdict into
combined_acquisitions (INSERT OR REPLACE - these two stages are deterministic
per dwhId given the current ES state, so a rerun is meant to overwrite).

Stage 3 (S3 web scrape) reuses s3_detection.py as-is, unmodified - including
its own corroboration/conflict logic, which now runs against the stage 1+2
row this script just wrote instead of whatever was already in the DB before.

Input file: one dwhId per row in the first column, no header expected.

Usage:
    python run_pipeline_on_file.py "input.csv" [db_path] [output.xlsx]
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sqlite3 import Row, connect
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

from combined_check import classify_review_status  # noqa: E402
from test_combined_pipeline import _combine_results  # noqa: E402
from s3_detection import (  # noqa: E402
    S3_BUCKET_NAME,
    S3_PREFIX,
    _create_table_if_missing,
    _s3_client,
    combined_text_blob_multi,
    extract_acquirer_from_s3_text,
    fetch_all_scrapes,
    list_pipeline_folders,
    merge_s3_result_into_db,
)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

MAX_WORKERS = 10

DEFAULT_DB_PATH = "test_combined_pipeline.db"
DEFAULT_OUTPUT_XLSX = "full_pipeline_output.xlsx"

OUTPUT_COLUMNS = [
    "dwhId", "companyName", "isAcquired", "acquirerCompany", "source",
    "matchedPattern", "confidenceScore", "reviewStatus",
    "nameFound", "descriptionFound", "descriptionAcquirer", "s3Found", "logoFound",
]


def _read_dwh_ids(path: str) -> list[str]:
    read = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = read(path, header=None, dtype=str)
    return [v.strip() for v in df.iloc[:, 0].dropna().tolist() if v.strip()]


# --- Stage 1 + 2: companyName + description (ES-sourced) --------------------


def _run_stage12(dwh_id: str, data_source: CompanyDataSource) -> dict[str, Any]:
    """Never raises - failures come back as isAcquired=None instead of dropping.

    nameFound/descriptionFound/descriptionAcquirer record what EACH detector
    independently saw, alongside (not instead of) the existing merged verdict
    from _combine_results() - _combine_results itself is untouched, this only
    exposes information it already computes internally so the stage-4 all-
    agree override (see logo_detection.py) has something to check later."""
    try:
        record = data_source.get_by_dwh_id(dwh_id)
        if record is None:
            return {
                "dwhId": dwh_id, "companyName": None, "isAcquired": None,
                "acquirerCompany": None, "source": None, "matchedPattern": None,
                "confidenceScore": None, "reviewStatus": None,
                "nameFound": None, "descriptionFound": None, "descriptionAcquirer": None,
            }
        name_match = check_company_record(record)
        description_match = extract_acquirer_from_description(record.raw.get("description"))
        combined = _combine_results(name_match, description_match, record.raw)
        # _combine_results() never sets reviewStatus (pre-existing gap in this
        # script, unrelated to the nameFound/descriptionFound additions below -
        # _write_stage12_row has always read row["reviewStatus"], so this was
        # a latent KeyError waiting to be hit). Derive it the same way every
        # other stage in this pipeline already does (s3_detection.py,
        # logo_detection.py) via classify_review_status - but keep
        # _combine_results' own isAcquired value as-is rather than letting
        # classify_review_status's score-threshold isAcquired override it,
        # since that determination belongs to the untouched detection logic.
        _, review_status = classify_review_status(combined.get("confidenceScore"), combined.get("matchedPattern"))
        combined = {**combined, "reviewStatus": review_status}
        description_acquirer = (
            description_match["acquirerCompany"]
            if description_match is not None and description_match["acquirerCompany"] is not None
            else None
        )
        return {
            "dwhId": dwh_id, "companyName": record.company_name, **combined,
            "nameFound": name_match is not None,
            "descriptionFound": description_acquirer is not None,
            "descriptionAcquirer": description_acquirer,
        }
    except Exception as exc:
        print(f"  dwhId {dwh_id}: stage 1+2 failed - {exc}")
        return {
            "dwhId": dwh_id, "companyName": None, "isAcquired": None,
            "acquirerCompany": None, "source": None, "matchedPattern": None,
            "confidenceScore": None, "reviewStatus": None,
            "nameFound": None, "descriptionFound": None, "descriptionAcquirer": None,
        }


def _write_stage12_row(conn, row: dict[str, Any]) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO combined_acquisitions "
        "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus, "
        "nameFound, descriptionFound, descriptionAcquirer) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row["dwhId"], row["companyName"], row["isAcquired"], row["acquirerCompany"],
            row["source"], row["matchedPattern"], row["confidenceScore"], row["reviewStatus"],
            row["nameFound"], row["descriptionFound"], row["descriptionAcquirer"],
        ),
    )


# --- Stage 3: S3 web scrape ---------------------------------------------------


def _run_stage3(s3, folders: list[str], dwh_id: str) -> dict[str, Any]:
    """S3 fetch + extraction only - no DB access, same threading discipline
    as s3_detection.py's own process_dwh_id (DB writes happen single-
    threaded afterward)."""
    scrapes, found_in = fetch_all_scrapes(s3, S3_BUCKET_NAME, folders, dwh_id)
    if not scrapes:
        return {"dwhId": dwh_id, "companyName": None, "s3_result": None, "no_data": True}
    company_name = next((s.get("companyName") for s in scrapes if s.get("companyName")), None)
    blob = combined_text_blob_multi(scrapes)
    s3_result = extract_acquirer_from_s3_text(blob, company_name)
    return {"dwhId": dwh_id, "companyName": company_name, "s3_result": s3_result, "no_data": False}


def main(input_path: str, db_path: str, output_xlsx: str) -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    conn = connect(db_path)
    _create_table_if_missing(conn)

    # --- Stage 1 + 2 ---
    print("\nStage 1+2 (companyName + description)...")
    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)
    stage12_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_stage12, dwh_id, data_source): dwh_id for dwh_id in dwh_ids}
        done = 0
        for future in as_completed(futures):
            stage12_results.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  stage 1+2: {done}/{len(dwh_ids)}")

    for row in stage12_results:
        _write_stage12_row(conn, row)
    conn.commit()

    stage12_acquired = sum(1 for r in stage12_results if r["isAcquired"] == 1)
    print(f"Stage 1+2 done: acquired={stage12_acquired} of {len(dwh_ids)}")

    # --- Stage 3 (corroborates/conflicts against the stage 1+2 rows just written) ---
    print("\nStage 3 (S3 web scrape)...")
    s3 = _s3_client()
    folders = list_pipeline_folders(s3, S3_BUCKET_NAME, S3_PREFIX)

    stage3_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_run_stage3, s3, folders, dwh_id): dwh_id for dwh_id in dwh_ids}
        done = 0
        for future in as_completed(futures):
            stage3_results.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  stage 3: {done}/{len(dwh_ids)}")

    stage3_outcomes: dict[str, int] = {}
    for row in stage3_results:
        if row["no_data"]:
            stage3_outcomes["no_s3_data"] = stage3_outcomes.get("no_s3_data", 0) + 1
            continue
        outcome = merge_s3_result_into_db(conn, row["dwhId"], row["companyName"], row["s3_result"])
        stage3_outcomes[outcome] = stage3_outcomes.get(outcome, 0) + 1
        # s3Found records whether THIS stage independently found an acquirer -
        # separate from merge_s3_result_into_db's own corroborate/conflict
        # verdict above, which is untouched. Only set when stage 3 actually
        # ran (had scrape data) - no_data rows above leave it NULL, not False.
        conn.execute(
            "UPDATE combined_acquisitions SET s3Found = ? WHERE dwhId = ?",
            (row["s3_result"] is not None, row["dwhId"]),
        )
    conn.commit()
    print("Stage 3 done:", stage3_outcomes)

    # --- Export final state (all 3 stages merged) ---
    conn.row_factory = Row
    placeholders = ",".join("?" * len(dwh_ids))
    db_rows = {
        r["dwhId"]: dict(r)
        for r in conn.execute(
            f"SELECT * FROM combined_acquisitions WHERE dwhId IN ({placeholders})", dwh_ids
        ).fetchall()
    }
    conn.close()

    order = {dwh_id: i for i, dwh_id in enumerate(dwh_ids)}
    output_rows = [db_rows.get(dwh_id, {"dwhId": dwh_id}) for dwh_id in dwh_ids]
    output_rows.sort(key=lambda r: order.get(r.get("dwhId"), len(dwh_ids)))

    df_out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    df_out.to_excel(output_xlsx, index=False)

    acquired = (df_out["isAcquired"] == 1).sum()
    likely_acquired = (df_out["reviewStatus"] == "likely_acquired").sum()
    print(f"\nWrote {len(df_out)} rows to {output_xlsx}")
    print(f"  acquired={acquired} likely_acquired={likely_acquired} not_acquired={len(df_out) - acquired - likely_acquired}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python run_pipeline_on_file.py "input.csv" [db_path] [output.xlsx]')
        sys.exit(1)
    _input_path = sys.argv[1]
    _db_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB_PATH
    _output_xlsx = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_XLSX
    main(_input_path, _db_path, _output_xlsx)
