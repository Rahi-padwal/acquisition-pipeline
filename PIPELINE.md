### TL;DR

For each company in the dataset, this pipeline answers: **Was this company acquired? By whom? How confident are we?**

It does this entirely with **rules,** regex pattern matching, cross-source corroboration, and hand-tuned confidence scoring. There is no machine learning or LLM anywhere in the decision path.

**Pipeline stages, at a glance:**

```
4 independent detectors → merge & score → full 768K-row batch run → false-positive audit
```

### 1. Purpose

Given a company record, the pipeline produces two outputs:

1. isAcquired - was this company acquired? (yes/no)
2. acquirerCompany + confidenceScore - if acquired, by whom, and how sure are we?

Confidence is not a probability estimate from a model, it's a number assigned to a specific matched pattern (e.g., `"X, a subsidiary of Y"` always starts at 1.0), then adjusted by a small set of explicit rules (agreement between sources, corroboration, plausibility checks).

### 2. Pipeline Flow

| Stage | What happens |
| --- | --- |
| **1. Detection** | Four independent detectors each scan a different data source (company name, description, scraped website text, logo image) to check if they are acquired or not |
| **2. Merge & Score** | Results are combined into one verdict per company, with a single confidence score, using cross-source agreement/disagreement rules. |
| **3. Full Batch Run** | The pipeline runs across all ~800K companies. |
| **4. False-Positive Audit** | A second pass re-examines every row already marked "acquired," correcting verdicts the scoring logic likely got wrong. |

### 3. The Four Detectors

All four detectors look for the **same families of acquisition phrasing -** "acquired by," "a subsidiary of," "part of,", etc. - just applied to different text. Every match is tagged with a matched_pattern name (e.g. `subsidiary_of`, `acquired_by`) that travels through the rest of the pipeline and drives scoring.

| Detector | Source file | What it scans |
| --- | --- | --- |
| **Name / structured fields** | `name_matching/check_name_signals.py` | Company name string + Elasticsearch's own `isAcquired`/`acquiredBy` fields |
| **Description text** | `description_matching/description_detection_api.py` | Company description |
| **Scraped website text** | `combined_matching/s3_detection.py` | Prose from About pages, Terms of Service, privacy pages (stored in S3) |
| **Logo OCR text** | `logo_matching/detection.py` | Text extracted from logo images via OCR |

### 4. Scoring & Merging Logic

`combined_matching/combined_check.py` is the single place where name-based and description-based results meet for the same company - this is where all cross-source logic lives.

**Rules:**

| Situation | Effect |
| --- | --- |
| Both sources **agree** | +0.05 bonus, capped at 0.98. |
| Both sources **disagree** | Capped at 0.4 (or 0.7 if the pattern is already "high-trust," base score ≥ 0.9). |
| ES structured field populated, text score low/middling | Score nudged up - manual review found the raw ES field is reliable even without text corroboration. |

#### Final bucketing (`classify_review_status()`)

Every row lands in one of five outcomes:

| Bucket | Meaning |
| --- | --- |
| `acquired` | Confident acquisition |
| `likely_acquired` | Score ≤ 0.35 "review floor" — weak but not worthless, kept for human review rather than silently dropped |
| `not_acquired` | No credible signal |
| `within_group` | Acquired, but the pattern describes group membership specifically |
| `subsidiary` | Acquired, pattern describes a subsidiary relationship specifically (e.g. `division_of`) |

### 5. The False-Positive Audit Layer

A **second, later pass** over every row already marked `isAcquired = 1`, specifically hunting for false positives.

#### Why this exists

An earlier approach tried counting how many *other* sources independently corroborated the on-record acquirer name, flagging a low count as suspicious. Across two separate rounds of manual review, this kept producing real counter-examples **in both directions**:

- Genuine acquisitions with zero corroboration
- Non-acquisitions with high corroboration

So corroboration-counting was abandoned in favor of a small set of **specific, narrow rules**, each justified by a documented manual-review finding.

#### The rules

1. **Implausible acquirer names → rejected.** Garbage captures, purely generic business words with no distinguishing name (bare "Group," "Capital Partners"), or the company's own name reflected back.
2. **Short, single-word names from scraped text → rejected.** A 1–3 letter fragment pulled from long unstructured prose is almost always a broken capture, not a real company name.
3. **Truncated-name recovery.** Some scraped text contains an abbreviation period (e.g. "N. Harris Computer Corporation") where the capture regex deliberately stops early — to avoid swallowing an unrelated sentence — leaving a fragment like "N." Before judging plausibility, the audit re-scans the original text to try to recover the fuller name.
4. **Live ES override.** If the company's *current* Elasticsearch record still confirms the acquisition, the row is **never** flagged as false positive — regardless of what the other rules say. The live field is treated as more authoritative than a text-corroboration guess.
5. **Manually confirmed exceptions.** A short list of individually verified cases that no general rule could safely cover.

#### Net effect

A company is only flagged as a false positive if **all** of these hold:

- Originally marked acquired
- Live ES record doesn't confirm it
- No manual override applies
- At least one specific rule (division_of, implausible name, or short scraped-text name) applies

### 6. Results Summary

From `combined_matching/combined_with_ocr.db` (`combined_acquisitions` table):

| Metric | Value |
| --- | --- |
| Total companies processed | 851,626 |
| Flagged `isAcquired = 1` | 17,692 |
| Flagged `isAcquired = 0` | 833,934 |
| Rows sourced from S3 scrape detection specifically | 2,586 |

Note: the finer-grained `reviewStatus` bucketing (`acquired` / `likely_acquired` / `not_acquired` / `within_group` / `subsidiary`) is only backfilled for a subset of rows (~14K) so far, not the full dataset — the numbers above use the coarser `isAcquired` flag, which is populated for every row.

### 7. Known Limitations

- **OCR accuracy on logo images is inconsistent.** Company logos are scraped from LinkedIn at low resolution/compression, which limits OCR text-extraction accuracy. Manual review of the logo-OCR detector's output found a meaningful proportion of extracted "acquirer" text was garbled or nonsensical, even after filtering.
- **Logo OCR needed hand-tuned workarounds for the production instance's CPU architecture.** On this machine (AWS Graviton2/Neoverse-N1), PaddleOCR's defaults crash: multi-threaded matmul segfaults inside OpenBLAS (a known Graviton/OpenBLAS runtime-CPU-dispatch bug), and `mkldnn` acceleration hits an unrelated oneDNN/PIR-executor incompatibility. Both are worked around with fixed config (`cpu_threads=1`, `enable_mkldnn=False` — see `logo_matching/ocr.py`), so OCR does run on the production instance itself; it's just single-threaded per worker rather than multi-threaded, with parallelism coming from the thread pool instead.
- **Some acquisition phrasing patterns remain unvalidated at scale.** Most manual verification effort to date has focused on patterns already suspected to be weak (`powered_by`, `joined_forces`, `self_by_acquirer`). Higher-confidence patterns (`was_acquired_by`, `subsidiary_of`, `acquired_by`, `trading_as`, etc.) have been spot-checked but not exhaustively audited, so their true real-world precision at full scale is not yet independently confirmed.
- **The false-positive audit's rule set is not exhaustive.** It targets specific, previously observed failure patterns rather than every possible failure mode. New false-positive shapes may exist in the data that haven't yet been discovered through manual review.