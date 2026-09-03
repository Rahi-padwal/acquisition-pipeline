"""One-off manual verification script: run a fixed, hardcoded list of dwhIds
(known powered_by/joined_forces false-positive and true-positive cases from
the powered_by/joined_forces quality-filter work) through the current
detection pipeline and print the results as a table.

Reuses process_company() from test_combined_pipeline.py as-is - the exact
same companyName detection + description detection + merge logic
run_full_batch.py uses - rather than reimplementing any of it here.

Not part of the main pipeline: no SQLite, no threading, just console output.
Meant to be run once, read by a human, then discarded.

Run with: python test_known_bugs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_source import ElasticCompanyDataSource  # noqa: E402 (from name_matching)

from test_combined_pipeline import _safe_str, process_company  # noqa: E402

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

# dwhId -> expectation, from manual review of the powered_by/joined_forces
# quality-filter work. "expect match" = a genuine acquisition that should
# still come through after the new filters; "expect suppressed" = a known
# false positive (franchise network, accelerator/fund, or garbage-text
# capture) that the new filters should now discard entirely.
DWH_IDS = [
    "0a6a0483b2371761844cfcaf68e6646c",  # ProVerify - Application Verification - expect suppressed
    "16240d248aa8d960cbf12aacc8b745d8",  # Willux.be - AXI - expect match
    "1fc02791059a09e55ef6d29804d645d5",  # IT Provider - AXI - expect match
    "39cd86cd4b3b39dd11ba4fa4d6c68846",  # A&M nv - AXI - expect match
    "3ceb746655e6e2f7383cd15c0e0863eb",  # Baymare - Proforma - expect suppressed
    "3dcf4c98f448fae23a502d89b67e5ef9",  # Spikes - Axi - expect match
    "5859faa7a118b9353af2eb0b592901f5",  # Metro Printed Products - Proforma - expect suppressed
    "5c5896e8555f5f07d898b3ca44c6e67d",  # NC Tweener Fund - NC IDEA - expect suppressed
    "5e29eeacd89f9496d1f3527da28627fd",  # ProVerify dup - expect suppressed
    "6219ad146f2521b286d76842776b0063",  # The Branding Agency - Proforma - expect suppressed
    "652144ac31fc27ccfdafe2398abe0046",  # TBWA Indonesia - garbage text - expect suppressed
    "6f448ae40d9294afe83e4ab4f3757f0d",  # Mixing Board - Axios - VERIFY MANUALLY, uncertain
    "892128bbf3db6216cacb2ba03d63ad92",  # Jiraaf - AI Growth - expect suppressed
    "8a0d53bd8e780bcc70ec30f447da5f76",  # Core ICT NV - AXI - expect match
    "d0559284e121152d0ab61f3dcee60e27",  # DVIT - AXI - expect match
    "d821be25941fecb197c8da752fcb1522",  # BridgeWay Firm - garbage text - expect suppressed
    "dfc0acf35429cd84e80d5e44e8a141ab",  # INTERIT - AXI - expect match
    "f7cac26cd6a4c7f357a7915e6d604326",  # ADS Phoenix - Proforma - expect suppressed
]

_HEADERS = ["dwhId", "companyName", "isAcquired", "acquirerCompany", "source", "matchedPattern", "confidenceScore"]


def _print_table(rows: list[dict[str, Any]]) -> None:
    values = [[_safe_str(row.get(h)) for h in _HEADERS] for row in rows]
    widths = [max(len(_HEADERS[i]), *(len(v[i]) for v in values)) for i in range(len(_HEADERS))]

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(_HEADERS))
    print("-+-".join("-" * w for w in widths))
    for v in values:
        print(fmt_row(v))


def main() -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)

    rows = [process_company(dwh_id, data_source) for dwh_id in DWH_IDS]
    _print_table(rows)


if __name__ == "__main__":
    main()
