"""Single-command entry point for the FULL 4-stage acquisition-detection
pipeline: companyName -> description -> S3 web scrape -> logo OCR, in that
order, over a list of dwhIds read from a CSV/Excel file.

Pure orchestration wrapper - calls run_pipeline_on_file.main() (stages 1-3)
then logo_detection.main() (stage 4) in sequence, in the same process, with
no changes to either module's logic. Stage 4 must run second: it reads back
whatever stages 1-3 already wrote to `db_path` and corroborates/conflicts
against it (see logo_detection.py's merge_logo_result_into_db and its
final all-stages-agree confidence override).

Usage:
    python run_pipeline_with_logo.py "input.csv" [db_path] [stage123_output.xlsx] [stage4_output.xlsx]
"""

from __future__ import annotations

import sys

import logo_detection
import run_pipeline_on_file

DEFAULT_DB_PATH = "test_combined_pipeline.db"
DEFAULT_STAGE123_XLSX = "full_pipeline_output.xlsx"
DEFAULT_STAGE4_XLSX = "logo_detection_output.xlsx"


def main(input_path: str, db_path: str, stage123_xlsx: str, stage4_xlsx: str) -> None:
    print("=== Stages 1-3: companyName -> description -> S3 web scrape ===")
    run_pipeline_on_file.main(input_path, db_path, stage123_xlsx)

    print("\n=== Stage 4: logo OCR ===")
    logo_detection.main(input_path, db_path, stage4_xlsx)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            'Usage: python run_pipeline_with_logo.py "input.csv" '
            "[db_path] [stage123_output.xlsx] [stage4_output.xlsx]"
        )
        sys.exit(1)
    _input_path = sys.argv[1]
    _db_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB_PATH
    _stage123_xlsx = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_STAGE123_XLSX
    _stage4_xlsx = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_STAGE4_XLSX
    main(_input_path, _db_path, _stage123_xlsx, _stage4_xlsx)
