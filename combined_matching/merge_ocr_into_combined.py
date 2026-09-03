"""One-off: replay stage-4's own merge_logo_result_into_db() (unmodified,
imported from logo_detection.py) against the ALREADY-COMPUTED logo_ocr_results
rows from the just-finished full OCR batch, writing into a COPY of
combined_acquisitions_full.db (the original production db is never touched).

Does not re-fetch or re-run OCR - logo_ocr_results already has everything
extract_acquirer_from_logo_text would have returned (acquirerCompany,
matchedPattern, confidenceScore, rawText), so this just replays the exact
same DB-merge step logo_detection.main() would have run, including the
final all-required-stages-agree override block (copied verbatim from
logo_detection.py's main(), since that block isn't factored into its own
function) - which correctly never fires here since run_full_batch.py's
production rows never set nameFound/s3Found (those columns don't exist
until _ensure_stage_tracking_columns adds them as NULL below), same as it
would if this exact input were run through logo_detection.py itself.
"""
from __future__ import annotations

import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path("/home/ubuntu/acquired_company")
for sub in ("combined_matching", "name_matching", "description_matching", "logo_matching"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from logo_detection import merge_logo_result_into_db, _create_table_if_missing  # noqa: E402
from combined_check import classify_review_status, _company_identity_match  # noqa: E402

SRC_DB = ROOT / "combined_matching" / "combined_acquisitions_full.db"
WORK_DB = ROOT / "combined_matching" / "combined_with_ocr.db"

print(f"Copying {SRC_DB} -> {WORK_DB} ...")
shutil.copyfile(SRC_DB, WORK_DB)

conn = sqlite3.connect(WORK_DB)
_create_table_if_missing(conn)  # adds nameFound/descriptionFound/descriptionAcquirer/s3Found/logoFound if missing

src_conn = sqlite3.connect(f"file:{SRC_DB}?mode=ro", uri=True)
src_conn.row_factory = sqlite3.Row

cur = src_conn.execute(
    "SELECT dwhId, companyName, fetchOutcome, matchedPattern, acquirerCompany, confidenceScore, rawText "
    "FROM logo_ocr_results ORDER BY rowid"
)

outcomes: dict[str, int] = {}
processed = 0
merged = 0
t0 = time.time()
for r in cur:
    processed += 1
    if r["fetchOutcome"] != "ok":
        continue
    logo_result = None
    if r["acquirerCompany"]:
        logo_result = {
            "acquirerCompany": r["acquirerCompany"],
            "matchedPattern": r["matchedPattern"],
            "confidenceScore": r["confidenceScore"],
            "rawText": r["rawText"],
        }
    outcome = merge_logo_result_into_db(conn, r["dwhId"], r["companyName"], logo_result)
    outcomes[outcome] = outcomes.get(outcome, 0) + 1
    conn.execute(
        "UPDATE combined_acquisitions SET logoFound = ? WHERE dwhId = ?",
        (logo_result is not None, r["dwhId"]),
    )
    merged += 1
    if merged % 100000 == 0:
        print(f"  merged {merged} ({processed} scanned) in {time.time()-t0:.0f}s - {outcomes}")

conn.commit()
print(f"Merge done: scanned={processed} merged={merged} in {time.time()-t0:.0f}s - {outcomes}")

# --- Final aggregate: all-required-stages-agree confidence override ---
# Copied verbatim from logo_detection.py main() (lines ~400-419) - same
# requirement (nameFound AND s3Found AND logoFound), same description
# carve-out. Included for consistency even though it will not fire here:
# run_full_batch.py's production rows never populated nameFound/s3Found
# (this run's stage 1-3 pass predates that per-stage tracking), so this
# is an honest no-op, not a fabricated result.
conn.row_factory = sqlite3.Row
all_rows = conn.execute("SELECT * FROM combined_acquisitions").fetchall()
overridden = 0
for r in all_rows:
    if not (r["nameFound"] and r["s3Found"] and r["logoFound"]):
        continue
    _, review_status = classify_review_status(1.0)
    conn.execute(
        "UPDATE combined_acquisitions SET isAcquired = 1, confidenceScore = 1.0, reviewStatus = ? WHERE dwhId = ?",
        (review_status, r["dwhId"]),
    )
    overridden += 1
conn.commit()
print(f"All-stages-agree override applied to {overridden} dwhId(s)")

total_rows = conn.execute("SELECT COUNT(*) FROM combined_acquisitions").fetchone()[0]
print(f"Final combined_acquisitions row count: {total_rows}")
conn.close()
src_conn.close()
