"""FastAPI wrapper around check_text_signals.run_check().

Run with: python -m uvicorn api:app --reload --port 8001
Docs at:  http://127.0.0.1:8001/docs
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

from check_text_signals import derive_company_name, run_check

app = FastAPI(title="Acquisition Signal Checker")

# Playwright's sync API launches the browser via a subprocess. Uvicorn on
# Windows runs its worker threads under a SelectorEventLoop, which cannot
# create subprocesses (NotImplementedError). Running each check in its own
# process sidesteps that restriction entirely, regardless of uvicorn's
# event-loop policy.
_executor = ProcessPoolExecutor(max_workers=4)


class CheckRequest(BaseModel):
    company_url: HttpUrl


class Evidence(BaseModel):
    candidate_name: str
    source_url: str
    matched_sentence: str


class DebugLogEntry(BaseModel):
    url: str
    status: str
    reason: str | None


class CheckResponse(BaseModel):
    company_name: str
    is_acquired: bool
    confidence: str
    likely_acquirer: str | None
    evidence: list[Evidence]
    status: str
    debug_log: list[DebugLogEntry]


def _run_check_for_url(company_url: str) -> dict:
    """Runs in the worker process: derives the display name from the URL,
    then delegates to run_check(url, name)."""
    return run_check(company_url, derive_company_name(company_url))


@app.post("/check", response_model=CheckResponse)
async def check(request: CheckRequest) -> CheckResponse:
    """Fetch the company's site and report any acquisition signal found."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run_check_for_url, str(request.company_url))
    return CheckResponse(**result)


@app.get("/check", response_model=CheckResponse)
async def check_get(company_url: HttpUrl) -> CheckResponse:
    """Same as POST /check, exposed as GET query params for convenience."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run_check_for_url, str(company_url))
    return CheckResponse(**result)
