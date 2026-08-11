# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A tool for detecting whether a company has been acquired, by scraping its public website and matching factual acquisition language. Two entry points:

- **CLI**: `python check_text_signals.py <url> [company_name]`
- **API**: `python -m uvicorn api:app --reload --port 8001` — docs at `http://127.0.0.1:8001/docs`

## Running tests

```
python test_check_text_signals.py
```

The test file is a standalone runner (not pytest). Mocked tests run fast; live tests (`test_blinkit_live_parent_company`, `test_whimsical_live_not_acquired`, `run_live_samples`) hit real sites over the network.

## Architecture

**`check_text_signals.py`** is the core. Pipeline:

1. `get_tiered_target_urls()` builds candidate sub-URLs in three tiers: high-signal pages (press/about/newsroom), legal pages (terms/privacy), then the homepage last.
2. `run_check()` scans tier by tier and stops as soon as evidence is found — the homepage is only consulted if tiers 1–2 produced nothing.
3. From press/news listing pages only, `_extract_acquisition_leaning_links()` follows a bounded (≤15), keyword-filtered set of same-domain article links. This is not general crawling.
4. `fetch_page_with_status()` uses Playwright (headless Chromium) to render each page with a realistic UA/headers to avoid bot-protection challenges.
5. `extract_text()` pulls title, meta description, footer, and full body text via BeautifulSoup.
6. `check_acquirer_match()` runs `_STRICT_ACQUIRER_PATTERNS` (regex, case-sensitive) against each field. Every candidate must clear three gates: no hypothetical language nearby, no self-description nearby (for the "a X company" pattern), and the denylist/shape/self-reference check in `_passes_validation()`.

**`api.py`** is a thin FastAPI wrapper. On Windows, Playwright's sync API can't create subprocesses inside uvicorn's SelectorEventLoop, so each `run_check()` call is dispatched to a `ProcessPoolExecutor` worker.

## Key design constraints

**Strict matching is intentional.** The system prefers false negatives over false positives. The regex patterns, denylist, hypothetical-language filter, and generic-term denylist all exist to prevent the class of bugs (e.g., "Registration Data" flagged as an acquirer from WHOIS boilerplate) that a looser NLP approach caused.

**Confidence levels**: `_PATTERN_CONFIDENCE` maps each pattern to `"high"` (unambiguous ownership claims: subsidiary_of, was_acquired_by, parent_company, etc.) or `"medium"` (softer: backed_by, owned_by, a_x_brand, a_x_company, a_x_product). Multiple different candidate names from one check always yields `"unconfirmed"`.

**`debug_log`**: populated only when `status == "unreachable"`. Each entry records the URL, status, and failure reason (timeout / DNS failure / connection refused / etc.) so the caller knows exactly what was tried.
