"""One-time migration: backfill reviewStatus (and correct isAcquired) on a
combined_acquisitions table written before the review-bucket change existed.
Mirrors combined_check.classify_review_status()'s floor in SQL - keep the
two in sync if the floor ever changes. Pure recompute from the confidenceScore
already stored, no ES calls, so this is instant even against the full
768k-row table - re-running the batch itself is NOT necessary for this.

Idempotent: safe to run more than once, always recomputes from
confidenceScore rather than accumulating adjustments.

Run with: python migrate_review_status.py [path/to/combined_acquisitions_full.db]
(defaults to combined_acquisitions_full.db in the current directory)
"""

from __future__ import annotations

import sqlite3
import sys

DEFAULT_DB_PATH = "combined_acquisitions_full.db"

# Mirrors combined_check._REVIEW_FLOOR - keep in sync.
_REVIEW_FLOOR = 0.35


def _ensure_review_status_column(conn: sqlite3.Connection) -> None:
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(combined_acquisitions)")}
    if "reviewStatus" not in existing_columns:
        conn.execute("ALTER TABLE combined_acquisitions ADD COLUMN reviewStatus TEXT")
        conn.commit()


def _print_counts(conn: sqlite3.Connection, label: str) -> None:
    rows = conn.execute(
        "SELECT reviewStatus, isAcquired, COUNT(*) FROM combined_acquisitions "
        "GROUP BY reviewStatus, isAcquired ORDER BY reviewStatus, isAcquired"
    ).fetchall()
    print(f"--- {label} ---")
    for review_status, is_acquired, count in rows:
        print(f"  reviewStatus={review_status!r:17} isAcquired={is_acquired!r:5} count={count}")


def migrate(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        _ensure_review_status_column(conn)
        _print_counts(conn, "before")

        # Error rows (isAcquired IS NULL, from a row that failed processing
        # entirely) are left untouched - reviewStatus stays NULL, same as before.
        conn.execute(
            "UPDATE combined_acquisitions SET reviewStatus = 'not_acquired' "
            "WHERE isAcquired IS NOT NULL AND confidenceScore IS NULL"
        )
        conn.execute(
            "UPDATE combined_acquisitions SET isAcquired = 0, reviewStatus = 'needs_review' "
            "WHERE confidenceScore IS NOT NULL AND confidenceScore <= ?",
            (_REVIEW_FLOOR,),
        )
        conn.execute(
            "UPDATE combined_acquisitions SET isAcquired = 1, reviewStatus = 'acquired' "
            "WHERE confidenceScore IS NOT NULL AND confidenceScore > ?",
            (_REVIEW_FLOOR,),
        )
        conn.commit()

        _print_counts(conn, "after")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB_PATH)
