"""FastAPI wrapper around combined_check.run_combined_check().

Runs name_matching's checks first, falling back to description_matching's
phrase-pattern detection only when the name-based check finds nothing.

Run with: python -m uvicorn api:app --reload --port 8004
Docs at:  http://127.0.0.1:8004/docs
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

from combined_check import run_combined_check  # noqa: E402
from data_source import CompanyDataSource, ElasticCompanyDataSource, StubCompanyDataSource  # noqa: E402 (from name_matching)

app = FastAPI(title="Combined Acquisition Checker")

_data_source: CompanyDataSource
if os.environ.get("ES_HOST"):
    _data_source = ElasticCompanyDataSource()
else:
    _data_source = StubCompanyDataSource()


class Evidence(BaseModel):
    candidate_name: str | None
    matched_pattern: str
    raw_name: str


class CheckResponse(BaseModel):
    company_id: str | None
    company_name: str | None
    is_acquired: bool
    likely_acquirer: str | None
    source: str | None
    evidence: list[Evidence]


@app.get("/check/{dwh_id}", response_model=CheckResponse)
async def check(dwh_id: str) -> CheckResponse:
    """Look up `dwh_id` and check it for an acquirer, name signals first,
    description phrase patterns as a fallback."""
    try:
        result = run_combined_check(dwh_id, _data_source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return CheckResponse(**result)
