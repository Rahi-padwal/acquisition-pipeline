"""FastAPI wrapper around check_name_signals.run_dwh_id_check().

Run with: python -m uvicorn api:app --reload --port 8002
Docs at:  http://127.0.0.1:8002/docs
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()  # picks up ES_HOST / ES_INDEX / ES_API_KEY from .env, if present

from check_name_signals import run_dwh_id_check
from data_source import CompanyDataSource, ElasticCompanyDataSource, StubCompanyDataSource

app = FastAPI(title="Name-Based Acquisition Checker")

# Real Elasticsearch if ES_HOST is set, else fall back to the in-memory stub.
_data_source: CompanyDataSource
if os.environ.get("ES_HOST"):
    _data_source = ElasticCompanyDataSource()
else:
    _data_source = StubCompanyDataSource()


class Evidence(BaseModel):
    candidate_name: str
    matched_pattern: str
    raw_name: str


class CheckNameResponse(BaseModel):
    company_id: str | None
    company_name: str
    is_acquired: bool
    likely_acquirer: str | None
    confidence_score: float | None
    confidence_notes: str | None
    evidence: list[Evidence]


@app.get("/check-name", response_model=CheckNameResponse)
async def check_name(dwh_id: str) -> CheckNameResponse:
    """Look up `dwh_id` (dwhId) in Elastic and check it for an acquirer."""
    result = run_dwh_id_check(dwh_id, _data_source)
    return CheckNameResponse(
        company_id=result["company_id"],
        company_name=result["company_name"],
        is_acquired=result["is_acquired"],
        likely_acquirer=result["likely_acquirer"],
        confidence_score=result.get("confidence_score"),
        confidence_notes=result.get("confidence_notes"),
        evidence=[Evidence(**e) for e in result["evidence"]],
    )
