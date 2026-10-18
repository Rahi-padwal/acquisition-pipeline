"""Standalone false-positive audit for a batch of dwhIds already marked
isAcquired=1 in combined_with_ocr.db.

Independent of confidence_scoring.py - does not import or modify it, and
changes nothing in the existing pipeline (no writes to combined_with_ocr.db).
Reimplements the same multi-source corroboration idea confidence_scoring.py
uses (does the acquirer name already on record for a company independently
show up in the raw text of the OTHER detection stages: companyName,
description, S3-scraped website text, logo OCR text?), reusing that script's
proven text-matching/S3/ES helpers by import, but deliberately differs from
it in three ways:

  1. S3 corroboration is always fetched and checked - no skip flag.
     confidence_scoring.py currently runs with SKIP_S3=True, which means a
     row whose only OTHER corroborating source is S3 loses that corroboration
     entirely, understating its real confidence.
  2. structured_field-sourced rows get NO free pass to confidence 1.0 - they
     go through the exact same 4-source corroboration check as every other
     row. confidence_scoring.py hands these an unconditional 1.0. But
     combined_with_ocr.db's own fp_review table (earlier Groq-based spot
     review, 3,553 rows) shows structured_field is actually the LEAST
     reliable pattern when it IS checked (~51% false_positive rate: 826 FP
     vs 302 genuine reviewed) - sampling those FPs shows the ES structured
     field often just holds a generic/junk string, not a real acquirer name.
  3. The stage a row's acquirer name ORIGINALLY came from (combined_
     acquisitions.source) is excluded from its own corroboration count - see
     _origin_stages_to_exclude(). Checking whether an acquirer name appears
     in, say, companyName when it was extracted BY REGEX FROM companyName in
     the first place is circular, not independent evidence, and inflates the
     score. Confirmed as a real, consequential gap by testing: e.g.
     "PoshVine by Razorpay" (self_by_acquirer, a genuine, independently-
     corroborated acquisition) scored a trivial 1.0 with "name" counted as
     one of 3 "corroborating" sources before this exclusion was added.

On top of that corroboration score, several purely rule-based checks are layered
(no LLM, no reliance on Groq's per-row verdicts - those were independently
found unreliable by the user, and even the aggregate "generic acquirer"
reason tag in fp_review turned out to flag plenty of real companies -
Cisco, Wipro, AOL, Warburg Pincus, Permira - as "generic", so it is NOT used
here as a source of truth or as a blocklist seed):

  4. implausible acquirer name - reuses s3_detection.py's own
     _is_garbage_acquirer() (self-referential capture, no proper-noun-like
     token, bare legal-suffix-only, >6 words, or "acquir" substring leaking
     through), plus two additional narrow, structural checks defined here:
     the acquirer name is nothing but generic business-structure descriptor
     words (e.g. bare "Group" or "Capital Partners" with no distinguishing
     name), or the acquirer name IS the company's own name (self-reference).
  5. knownBadPattern - informational only (see "corroboration score is not
     used to decide FP" below for why this no longer drives the verdict on
     its own). Still surfaced in the output: the row's original
     matchedPattern is one the pipeline's own history/data shows is
     historically unreliable overall - self_by_acquirer (measured ~75%
     wrong per the project's own commit history when UNCORROBORATED - see
     _KNOWN_BAD_PATTERNS below - retired from the live detector but old
     rows never reclassified), acquirer_acquires_self (~80% FP in
     fp_review), provided_by (~62% FP in fp_review, already flagged risky
     in confidence_scoring.py's _RISKY_ORIGINAL_PATTERNS).

  6. ES-live-isAcquired override - if the company's OWN Elastic record has
     isAcquired truthy right now, the row is never flagged FP, regardless of
     what the corroboration/implausibility checks above say. Confirmed as a
     real, consequential gap by testing dwhId
     65e858fcab52723d4465d0944c152426 (Datavail, acquirer "CIVC Partners",
     matchedPattern=structured_field): this pattern only exists because
     check_name_signals.py's _check_structured_fields() saw ES's own
     isAcquired flag true and pulled the acquirer straight out of
     fundingAndInvestment.acquiredBy - not a name/description/S3/logo text
     guess. A private-equity acquirer name like this legitimately may never
     appear verbatim in the company's own scraped site text, so the
     corroboration check (built for regex-extracted-from-text patterns)
     scored it 0.0 and flagged FP=True. That's the corroboration heuristic
     failing to find evidence it was never going to find, not ES being
     wrong - ES's own field is the more authoritative signal here and wins.

  7. division_of pattern is a subsidiary description, not an acquisition
     event - forced FP=True (unless the ES override (6) above independently
     confirms a real acquisition record). Manual review finding (user
     hand-checked 63 FP-flagged rows, 2026-09-04): every division_of row
     inspected ("X, a division of Y") described an ongoing corporate
     structure, not a discrete acquisition transaction this project tracks
     - "division of" doesn't imply Y ever separately acquired X. Independent
     corroboration doesn't rescue these either: a subsidiary's own site
     routinely and correctly mentions its parent, which isn't evidence FOR
     an acquisition, just a restatement of the same non-acquisition fact.

  8. short, single-word acquirer name captured FROM S3-SCRAPED TEXT
     specifically (combined_acquisitions.source == "s3_scrape") - forced
     FP=True (unless the ES override (6) above applies). Manual review
     finding (user, 2026-09-04, second batch of 100 fresh rows): every
     s3_scrape-origin row inspected whose captured acquirer was a single
     word of 3 letters or fewer (e.g. a bare 2-3 letter fragment) was NOT a
     genuine acquisition. S3-scraped site text is long, unstructured prose
     - a regex capturing only a 1-3-letter fragment out of it is almost
     always a broken/truncated capture (a stray initialism, abbreviation,
     or leftover fragment), not a real company name. Deliberately scoped to
     s3_scrape only, not every source: this project's own combined_check.py
     already treats fields differently by how structured they are, and a
     short acquirer captured from the much shorter, more structured
     companyName/description fields is a different situation, untested
     here - see logo_matching's own _is_garbage_acquirer as an existing
     precedent for a source-specific plausibility check like this one.

Corroboration score (confidenceScore/matchedStages) is NOT used to decide
FP anymore, even though it's still computed and shown in the output for
context. It was, through two rounds of manual review (63 rows, then a fresh
100 rows, 2026-09-04): the very first cut of this script treated
confidenceScore <= 0.3 (0 or 1 independent corroborating stage) as
low-corroboration-therefore-FP, with a narrow trust carve-out for a
single S3-only match. Manual review then flipped a single match (score
0.3) to trusted-by-default, carving OUT logo-only matches and the
a_x_company pattern as the two exceptions that should stay untrusted. But
the very next round of manual review directly contradicted BOTH of those
carve-outs (a logo-only match for Confluence Discovery Technologies/Aclaris
Therapeutics, and an a_x_company match for Kilborn Consulting/TYLin, were
both genuine acquisitions) as well as the older known-bad-pattern carve-out
(a single S3-only match for Shipway/Unicommerce, self_by_acquirer, was also
genuine - consistent with this same file's PoshVine example above: a
known-bad pattern with real corroboration is real evidence, even from just
one source). On top of that, the majority (7 of the 10 counter-examples
found in that second round) had confidenceScore == 0.0 - i.e. NO
independent corroboration anywhere - and were still genuine acquisitions.
Across two independent rounds of ground-truth checking, every corroboration-
count-based rule tried (the 0-count default, and all three single-match
exceptions) turned up real counter-examples. That is strong evidence the
signal itself is too weak to safely gate FP on, not that one more exception
would fix it - so it's retired from the FP decision entirely, in favor of
the small set of specific, source/pattern-scoped checks below (4, 6, 7, 8)
that have each survived scrutiny so far.

  9. Recover a period-truncated acquirer name before checks 4/8 judge it.
     Root cause CONFIRMED by pulling the raw text behind the "Known open
     tension" case from an earlier draft of this file (dwhId
     4993bfb1f2e7930a338a7274a448575d, Usewalter, acquirer stored as the
     single letter "N", matchedPattern=division_of, source=s3_scrape): the
     actual S3-scraped text reads "...HOPEM as a division of N. Harris
     Computer Corporation has signed a buying and selling shares
     agreement..." - s3_detection.py's (and logo_matching/detection.py's)
     shared acquirer-capture regex deliberately excludes periods from the
     captured span (correctly avoiding swallowing an entire unrelated
     trailing sentence - see s3_detection.py's own comment on this), so it
     stopped dead at "N." This re-scans the SAME raw text already fetched:
     if the stored acquirer is immediately followed there by ". " and then
     1-4 more capitalized words ending at a genuine boundary, the longer
     recovered name is used for checks 4/8 instead of the bare fragment.
     Deliberately narrow (exact, immediate, whole-word prefix match only) -
     not a general re-detection pass, and it does not touch
     s3_detection.py's own regex (see check 9's code comment for why a
     blanket "allow periods" change there would be a mistake).

FP=True when the original row is isAcquired=1 AND NEITHER the ES override
(6) NOR the manually-confirmed-acquisition override (see
_CONFIRMED_ACQUISITION_OVERRIDES) applies, AND any of:
  - matchedPattern == "division_of" (check 7).
  - the (possibly check-9-recovered) acquirer name is implausible (check 4).
  - the (possibly check-9-recovered) acquirer is a short, single-word
    S3-scrape capture (check 8).

Two known exceptions no available rule-based signal can reach safely - see
_CONFIRMED_ACQUISITION_OVERRIDES for the full reasoning on each:
Usewalter (check 9 correctly recovers the real acquirer, but division_of
must stay a blanket rule for other rows in this exact batch that are
correctly NOT acquisitions) and Blueprint HR Software Solutions (the real
acquirer name isn't fully recoverable from any of the 4 fetched texts at
all - an OCR miss, not a truncation this project's regexes caused).

Usage:
    python audit_false_positives.py ["Untitled spreadsheet (1).xlsx"] [output.xlsx]
"""

from __future__ import annotations

import gc
import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from data_source import ElasticCompanyDataSource  # noqa: E402

from confidence_scoring import (  # noqa: E402 (reused as-is, not modified)
    _name_found_in_text,
    _normalize,
)
from s3_detection import (  # noqa: E402 (reused as-is, not modified)
    S3_BUCKET_NAME,
    S3_PREFIX,
    _is_garbage_acquirer,
    _s3_client,
    combined_text_blob_multi,
    fetch_all_scrapes,
    list_pipeline_folders,
)

ES_HOST = os.environ.get("ES_HOST", "https://prodelastic.growthpal.com/")
ES_API_KEY = os.environ.get("ES_API_KEY")
ES_INDEX = os.environ.get("ES_INDEX", "sellers_0726")

# CHUNK_SIZE bounds peak memory to roughly one chunk's worth of fetched
# ES/S3 text instead of all 10k+ rows' worth at once - added after a
# full-batch run on this machine (1.8GB RAM total, already under sustained
# memory pressure from other running processes for the whole duration of a
# multi-minute run) was killed twice: once by the kernel OOM killer, and
# again (after chunking to 250) by a system-wide low-memory guard, even
# though the chunked run itself never spiked - the constraint here is total
# system memory over time, not just this process's own peak. S3_WORKERS is
# also dropped from confidence_scoring.py's 64. main() also checkpoints
# after every chunk and can resume from that checkpoint (see main()), so a
# kill partway through only costs the current chunk, not the whole run.
CHUNK_SIZE = 100
# Bumped ~3x (2026-09-04, user request to speed up the full 10,771-row run)
# from the original 5/8 - checked free memory first (400MB free, swap
# already at 923MB in use) and this is a moderate increase, not the ~9x
# that would be needed to hit a 10-minute target, which risks reproducing
# the OOM kill documented above. Revert to 5/8 if this run gets killed too.
ES_WORKERS = 15
S3_WORKERS = 24

SOURCE_DB_PATH = str(Path(__file__).resolve().parent / "combined_with_ocr.db")
BASELINE_OUTPUT_XLSX = str(Path(__file__).resolve().parent / "confidence_scoring_output.xlsx")

DEFAULT_INPUT_XLSX = "Untitled spreadsheet (1).xlsx"
DEFAULT_OUTPUT_XLSX = "audit_false_positives_output.xlsx"

_STAGE_PRIORITY = ["name", "description", "s3", "logo"]

# --- Corroboration guard for short acquirer names -----------------------------
#
# confidence_scoring._name_found_in_text's plain substring check
# (`needle in haystack`) has no minimum-length guard - only its word-level
# FUZZY fallback does (_MIN_FUZZY_LEN = 4 there). Confirmed as a real,
# consequential bug by testing: a 1-char acquirer name like "S" (a truncated
# OCR/regex extraction artifact - PaddleOCR read "An AS Company" and split
# the real 2-letter acronym "AS" into "A" + "S") scored a fake 1.0
# confidence (3/3 stages "corroborating") purely because the letter "s"
# trivially occurs somewhere in almost any English text (e.g. inside
# "Solutions"), not because of any real independent evidence.
#
# Reusing _name_found_in_text unmodified (not editing confidence_scoring.py)
# but gating it here: for a needle shorter than the same _MIN_FUZZY_LEN
# threshold, require it to appear as a distinct WHOLE WORD in the haystack
# instead of anywhere as a substring. This still lets a genuine short
# acronym corroborate correctly (e.g. "BD" for Becton Dickinson, confirmed
# elsewhere in this batch as a real, correct hit) when the text actually
# contains that word as its own token, while rejecting the spurious
# inside-another-word matches that caused the bug above.
_MIN_SUBSTRING_NEEDLE_LEN = 4


def _corroborates(acquirer: str | None, text: str | None) -> bool:
    if not acquirer or not text:
        return False
    needle = _normalize(acquirer)
    if not needle:
        return False
    if len(needle) < 2:
        # A single character can never be meaningful independent evidence.
        # Confirmed as a real, consequential residual gap by testing: even
        # after the whole-word fix below, a 1-char needle like "s" still
        # spuriously matched a real row in this batch - _normalize strips
        # punctuation, so an ordinary English possessive/contraction
        # ("Momentum's clients", "The firm's Technology") turns into stray
        # standalone 1-char tokens ("momentum s", "firm s") in the
        # haystack, with nothing to do with any acquirer name.
        return False
    if len(needle) < _MIN_SUBSTRING_NEEDLE_LEN:
        return needle in set(_normalize(text).split())
    return _name_found_in_text(acquirer, text)

# --- Check 4: known-bad matchedPattern values ---------------------------------
#
# self_by_acquirer: this project's own commit history measured it ~75% wrong
# and removed it from the live name_matching detector - old rows produced by
# it before that change were never reclassified.
# acquirer_acquires_self / provided_by: measured directly off
# combined_with_ocr.db's fp_review table (see module docstring) at ~80% and
# ~62% false_positive respectively.
#
# IMPORTANT: combined_check.py's own docstring on this notes the ~75% wrong
# figure for self_by_acquirer was measured on UNCORROBORATED rows
# specifically ("genuine hits get lifted above this floor already, by the
# corroboration bonus ... what's LEFT sitting at exactly 0.3 is
# disproportionately noise") - i.e. this was always a reason to distrust
# these patterns when corroboration is weak, not a reason to override
# strong independent corroboration when it exists - confirmed directly by
# testing self_by_acquirer rows like "PoshVine by Razorpay" and
# "Shipway by Unicommerce" (both genuine, single-source-corroborated
# acquisitions). This flag is informational only now (see "corroboration
# score is not used to decide FP" in the module docstring) - it no longer
# affects the FP verdict on its own.
_KNOWN_BAD_PATTERNS = {"self_by_acquirer", "acquirer_acquires_self", "provided_by"}

# --- Check 7: division_of pattern is a subsidiary description ----------------
#
# See check 7 in the module docstring for the full rationale. "X, a division
# of Y" describes an ongoing corporate structure, not a discrete acquisition
# transaction - every division_of row in the first reviewed batch was a
# subsidiary, not a genuine acquisition event. Kept as a blanket rule despite
# one confirmed counter-example (Usewalter, see _CONFIRMED_ACQUISITION_
# OVERRIDES below) because this same rule is independently confirmed correct
# elsewhere in this exact batch (VantageLinks, Equity Engineering Group) on
# rows with equally plausible-looking acquirer names - there's no available
# text signal that tells a genuine division_of acquisition apart from a
# non-acquisition one, so the one known exception is handled as an explicit
# override instead of weakening the rule for everyone.
_DIVISION_OF_PATTERN = "division_of"

# --- Manually-confirmed acquisition overrides ---------------------------------
#
# A small, explicit list of dwhIds independently confirmed as genuine
# acquisitions by external research (user, 2026-09-04), for the specific
# cases where no available rule-based signal reaches the same answer safely:
#
#   - Usewalter: check 9 correctly recovers the truncated acquirer as
#     "N. Harris Computer Corporation" (real, confirmed), but matchedPattern
#     is division_of - a rule that must stay in place for other rows (see
#     check 7's comment above), and division_of alone can't otherwise be told
#     apart from a genuine acquisition phrased the same way.
#   - Blueprint HR Software Solutions: the real acquirer ("HR Path Group")
#     isn't fully recoverable from any of the 4 fetched texts - the logo OCR
#     text dropped the distinguishing words ("...an HR company", missing
#     "Path"/"Group"), and the ES description only echoes a vaguer,
#     unconfirmed fragment ("an HR path company"). No text-based recovery
#     rule can safely bridge that gap.
#
# Same mechanism/precedent as the ES-live-isAcquired override (check 6) -
# just sourced from external verification instead of Elasticsearch. Add to
# this dict (not a code change elsewhere) as further confirmed exceptions
# turn up that no general rule can safely cover.
_CONFIRMED_ACQUISITION_OVERRIDES = {
    "4993bfb1f2e7930a338a7274a448575d": "Usewalter - acquired by N. Harris Computer (Constellation Software), Oct 2021",
    "38d9b4e7c6ebcfa07f92fdc8030007cb": "Blueprint HR Software Solutions - acquired by HR Path Group, Dec 2025",
}

# --- Origin-stage exclusion ----------------------------------------------------
#
# combined_acquisitions.source records which stage ORIGINALLY produced the
# acquirer name on file. Checking that same stage's raw text for the same
# name is circular - of course a name extracted BY REGEX FROM companyName
# also "appears in" companyName - so that stage is excluded from its own
# row's corroboration count; only the OTHER (genuinely independent) stages'
# text counts as corroboration. matchedPattern == "structured_field" is a
# deliberate exception: despite `source` recording "company_name" for these
# rows in the DB (a known mislabeling - see combined_check.py's own
# docstring: "a structured_field match came from ES's own isAcquired/
# acquiredBy fields, never from parsing the companyName string, and
# mislabeling it as company_name is misleading"), the acquirer name for a
# structured_field row actually comes from a separate ES field
# (fundingAndInvestment.acquiredBy.acquirerCompanyName), genuinely
# independent of the companyName text - so nothing is excluded for these.
_ORIGIN_STAGE_BY_SOURCE: dict[str, set[str]] = {
    "company_name": {"name"},
    "description": {"description"},
    "s3_scrape": {"s3"},
    "logo_ocr": {"logo"},
    "both": {"name", "description"},
}


def _origin_stages_to_exclude(source: str | None, matched_pattern: str | None) -> set[str]:
    if matched_pattern == "structured_field":
        return set()
    return _ORIGIN_STAGE_BY_SOURCE.get(source or "", set())

# --- Check 3: bare business-structure-descriptor acquirer names ---------------
#
# Deliberately narrow and NOT derived from fp_review's Groq "generic acquirer"
# reason tag (see module docstring - that tag itself flagged real companies
# like Cisco/Wipro/AOL as "generic", so it's not a trustworthy seed). Only
# flags an acquirer name that is ENTIRELY generic business-type words with no
# distinguishing proper noun at all (e.g. bare "Group", "Capital Partners") -
# a name that is legally just a business-structure descriptor can't itself be
# a real, identifiable acquiring company. A single plausible brand word (e.g.
# "Synergy", "Polaris") is deliberately NOT flagged, since it could be real.
_BARE_DESCRIPTOR_WORDS = {
    "group", "solutions", "holdings", "holding", "capital", "ventures",
    "enterprises", "consulting", "industries", "systems", "technologies",
    "services", "global", "worldwide", "partners", "collective",
    "network", "networks", "media", "digital", "labs", "company", "co",
}


def _is_bare_descriptor_acquirer(acquirer: str) -> bool:
    words = re.findall(r"[A-Za-z]+", acquirer.lower())
    if not words:
        return False
    return all(w in _BARE_DESCRIPTOR_WORDS for w in words)


def _is_self_reference_acquirer(acquirer: str, company_name: str | None) -> bool:
    a, c = _normalize(acquirer), _normalize(company_name)
    return bool(a) and bool(c) and a == c


def _is_implausible_acquirer(acquirer: str | None, company_name: str | None) -> bool:
    if not acquirer:
        return False
    stripped = acquirer.strip()
    if len(stripped) < 3:
        return True
    if _is_garbage_acquirer(stripped):
        return True
    if _is_bare_descriptor_acquirer(stripped):
        return True
    if _is_self_reference_acquirer(stripped, company_name):
        return True
    return False


# --- Check 8: short, single-word acquirer captured from S3-scraped text ------
#
# See check 8 in the module docstring for the full rationale - scoped
# specifically to combined_acquisitions.source == "s3_scrape" (checked by
# the caller), not applied to every source.
def _is_short_s3_scrape_acquirer(acquirer: str | None) -> bool:
    if not acquirer:
        return False
    words = re.findall(r"[A-Za-z]+", acquirer)
    return len(words) == 1 and len(words[0]) <= 3


# --- Check 9: recover a period-truncated acquirer name ------------------------
#
# See check 9 in the module docstring for the confirmed root cause: the
# acquirer-capture regexes in s3_detection.py and logo_matching/detection.py
# both deliberately exclude periods from the captured span, so a real
# acquirer name with an internal abbreviation period ("N. Harris Computer")
# gets cut off right there. Confirmed directly - Usewalter's S3 text reads
# "a division of N. Harris Computer Corporation has signed...", and the
# stored acquirer is exactly "N".
#
# This does not touch s3_detection.py's regex (a blanket "allow periods"
# change there reintroduces the swallow-the-next-sentence bug that exclusion
# was added to prevent - see that file's own comment on _RISKY_ACQUIRER_CAPTURE
# in name_matching/check_name_signals.py for the same failure mode). Instead
# it re-scans the SAME raw text already fetched for corroboration: if the
# stored acquirer name is immediately followed there by ". " and then 1-4
# more capitalized words, that longer span is almost certainly what a
# period-inclusive capture would have found. Deliberately narrow - requires
# the stored name to be an exact, immediate, whole-word prefix of the
# recovered span, not a general re-detection pass. No trailing punctuation
# lookahead is needed (unlike s3_detection.py's own _NAME_TERMINATOR): real
# prose continues with ordinary lowercase words right after a company name
# ("...Corporation has signed..."), not a comma/period, so the {0,3} cap on
# capitalized-word repetitions is what bounds the match instead - it simply
# stops adding words the moment a non-capitalized one comes next.
_TRUNCATED_CONTINUATION = re.compile(
    r"\.\s+([A-Z][\w&'-]*(?:\.?\s+[A-Z][\w&'-]*){0,3})"
)


def _recover_truncated_acquirer(acquirer: str | None, source_text: str | None) -> str | None:
    if not acquirer or not source_text:
        return None
    stripped = acquirer.strip()
    if not stripped or not stripped[-1].isalnum():
        return None
    for m in re.finditer(r"(?<!\w)" + re.escape(stripped) + r"(?!\w)", source_text):
        cont = _TRUNCATED_CONTINUATION.match(source_text, m.end())
        if cont:
            return f"{stripped}. {cont.group(1).strip()}"
    return None


# Maps combined_acquisitions.source to the one already-fetched raw text a
# truncated capture from that source would have come from - mirrors
# _ORIGIN_STAGE_BY_SOURCE's source->stage mapping above, minus the "both"
# case (name+description), which check_structured_fields-style truncation
# doesn't apply to anyway.
_SOURCE_TO_TEXT_KEY = {
    "company_name": "name",
    "description": "description",
    "s3_scrape": "s3",
    "logo_ocr": "logo",
}


def _read_dwh_ids(path: str) -> list[str]:
    df = pd.read_excel(path, dtype=str)
    col = "dwhId" if "dwhId" in df.columns else df.columns[0]
    return [v.strip() for v in df[col].dropna().tolist() if v.strip()]


def _load_existing(dwh_ids: list[str]) -> dict[str, dict[str, Any]]:
    conn = sqlite3.connect(SOURCE_DB_PATH)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(dwh_ids))

    existing: dict[str, dict[str, Any]] = {}
    q = (
        "SELECT dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern "
        f"FROM combined_acquisitions WHERE dwhId IN ({placeholders})"
    )
    for row in conn.execute(q, dwh_ids).fetchall():
        existing[row["dwhId"]] = {
            "originalCompanyName": row["companyName"],
            "originalIsAcquired": row["isAcquired"],
            "originalAcquirerCompany": row["acquirerCompany"],
            "originalSource": row["source"],
            "originalMatchedPattern": row["matchedPattern"],
        }

    q2 = f"SELECT dwhId, ocrText FROM logo_ocr_results WHERE dwhId IN ({placeholders})"
    for row in conn.execute(q2, dwh_ids).fetchall():
        existing.setdefault(row["dwhId"], {})
        existing[row["dwhId"]]["logoText"] = row["ocrText"]

    conn.close()
    return existing


def _run_es_stage(dwh_id: str, data_source: ElasticCompanyDataSource) -> dict[str, Any]:
    try:
        record = data_source.get_by_dwh_id(dwh_id)
    except Exception as exc:
        print(f"  dwhId {dwh_id}: ES fetch failed - {exc}")
        record = None

    if record is None:
        return {
            "dwhId": dwh_id, "companyName": None, "companyNameText": "", "descriptionText": "",
            "esIsAcquired": False,
        }

    return {
        "dwhId": dwh_id,
        "companyName": record.company_name,
        "companyNameText": record.company_name or "",
        "descriptionText": record.raw.get("description") or "",
        # ES's own live isAcquired field - see "ES-live-isAcquired override"
        # (check 6) in the module docstring. Independent of, and takes
        # priority over, this script's own text-corroboration score.
        "esIsAcquired": bool(record.raw.get("isAcquired")),
    }


def _run_s3_stage(s3, folders: list[str], dwh_id: str) -> dict[str, Any]:
    scrapes, _ = fetch_all_scrapes(s3, S3_BUCKET_NAME, folders, dwh_id)
    blob = combined_text_blob_multi(scrapes) if scrapes else ""
    return {"dwhId": dwh_id, "s3Text": blob}


def _score_company(
    es_row: dict[str, Any], s3_row: dict[str, Any], existing_row: dict[str, Any]
) -> dict[str, Any]:
    dwh_id = es_row["dwhId"]
    company_name = es_row["companyName"] or existing_row.get("originalCompanyName")
    original_acquired = bool(existing_row.get("originalIsAcquired"))
    original_pattern = existing_row.get("originalMatchedPattern")
    original_source = existing_row.get("originalSource")
    acquirer = existing_row.get("originalAcquirerCompany")
    es_confirmed_acquired = bool(es_row.get("esIsAcquired"))

    if not acquirer:
        return {
            "dwhId": dwh_id,
            "companyName": company_name,
            "acquirerCompany": None,
            "originalMatchedPattern": original_pattern,
            "confidenceScore": 0.0,
            "matchedStages": "",
            "implausibleAcquirer": False,
            "knownBadPattern": False,
            "shortS3Acquirer": False,
            "recoveredAcquirer": None,
            "esIsAcquired": es_confirmed_acquired,
            "manuallyConfirmedAcquired": dwh_id in _CONFIRMED_ACQUISITION_OVERRIDES,
            "FP": original_acquired and not es_confirmed_acquired and dwh_id not in _CONFIRMED_ACQUISITION_OVERRIDES,
        }

    texts = {
        "name": es_row["companyNameText"],
        "description": es_row["descriptionText"],
        "s3": s3_row["s3Text"],
        "logo": existing_row.get("logoText") or "",
    }
    excluded_stages = _origin_stages_to_exclude(original_source, original_pattern)
    independent_stages = [stage for stage in _STAGE_PRIORITY if stage not in excluded_stages]
    matched = [stage for stage in independent_stages if _corroborates(acquirer, texts[stage])]
    count = len(matched)

    if count >= 3:
        score = 1.0
    elif count == 2:
        score = 0.5
    elif count == 1:
        score = 0.3
    else:
        score = 0.0

    # Check 9 (see module docstring): try to recover a period-truncated
    # acquirer name from the same raw text it was originally captured from,
    # BEFORE checks 4/8 judge its plausibility - a truncated fragment like
    # "N" should be judged as "N. Harris Computer Corporation" if that's
    # what's actually sitting in the source text, not on its own.
    origin_text = texts.get(_SOURCE_TO_TEXT_KEY.get(original_source or ""))
    recovered_acquirer = _recover_truncated_acquirer(acquirer, origin_text)
    effective_acquirer = recovered_acquirer or acquirer

    implausible = _is_implausible_acquirer(effective_acquirer, company_name)
    known_bad_pattern = original_pattern in _KNOWN_BAD_PATTERNS  # informational only

    # Check 7 (see module docstring): division_of describes an ongoing
    # subsidiary relationship, not a discrete acquisition transaction -
    # forced FP=True regardless of corroboration score.
    is_division_of = original_pattern == _DIVISION_OF_PATTERN

    # Check 8 (see module docstring): a short, single-word acquirer name
    # captured specifically from S3-scraped text is almost always a broken
    # capture out of unstructured prose, not a real company name.
    short_s3_acquirer = original_source == "s3_scrape" and _is_short_s3_scrape_acquirer(effective_acquirer)

    # NOTE: corroboration count/score (computed above for the output columns)
    # is deliberately NOT part of this decision - see "corroboration score is
    # not used to decide FP" in the module docstring.
    #
    # Check 6 (ES-live-isAcquired override) and the manually-confirmed
    # override (see module docstring) both override every check below -
    # each is either the most authoritative signal available (ES) or a
    # specific, externally-verified exception (manual override).
    manually_confirmed_acquired = dwh_id in _CONFIRMED_ACQUISITION_OVERRIDES
    fp = (
        original_acquired
        and not es_confirmed_acquired
        and not manually_confirmed_acquired
        and (is_division_of or implausible or short_s3_acquirer)
    )

    return {
        "dwhId": dwh_id,
        "companyName": company_name,
        "acquirerCompany": acquirer,
        "originalMatchedPattern": original_pattern,
        "confidenceScore": score,
        "matchedStages": "+".join(matched),
        "implausibleAcquirer": implausible,
        "knownBadPattern": known_bad_pattern,
        "shortS3Acquirer": short_s3_acquirer,
        "recoveredAcquirer": recovered_acquirer,
        "esIsAcquired": es_confirmed_acquired,
        "manuallyConfirmedAcquired": manually_confirmed_acquired,
        "FP": fp,
    }


def _load_baseline(dwh_ids: list[str]) -> dict[str, bool]:
    """Prior confidence_scoring.py run's FP verdict per dwhId, for the diff
    column in the output - read-only, this file is never written to."""
    if not os.path.exists(BASELINE_OUTPUT_XLSX):
        return {}
    df = pd.read_excel(BASELINE_OUTPUT_XLSX, dtype={"dwhId": str})
    if "dwhId" not in df.columns or "FP" not in df.columns:
        return {}
    wanted = set(dwh_ids)
    return {
        row["dwhId"]: bool(row["FP"])
        for _, row in df.iterrows()
        if row["dwhId"] in wanted
    }


def _process_chunk(
    chunk: list[str],
    existing: dict[str, dict[str, Any]],
    baseline: dict[str, bool],
    data_source: ElasticCompanyDataSource,
    s3,
    folders: list[str],
) -> list[dict[str, Any]]:
    """Fetch ES + S3 text and score one bounded chunk of dwhIds, returning
    only the small scored-row dicts. The raw ES/S3 text for the chunk (the
    memory-heavy part - S3 scrape blobs especially, see s3_detection.py's
    own comment on capping fuzzy-match cost on huge blobs) goes out of scope
    when this call returns, instead of accumulating for all 10k+ rows at
    once - see CHUNK_SIZE's docstring for why this matters here."""
    es_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=ES_WORKERS) as executor:
        futures = {executor.submit(_run_es_stage, dwh_id, data_source): dwh_id for dwh_id in chunk}
        for future in as_completed(futures):
            row = future.result()
            es_results[row["dwhId"]] = row

    s3_results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=S3_WORKERS) as executor:
        futures = {executor.submit(_run_s3_stage, s3, folders, dwh_id): dwh_id for dwh_id in chunk}
        for future in as_completed(futures):
            row = future.result()
            s3_results[row["dwhId"]] = row

    output_rows = []
    for dwh_id in chunk:
        es_row = es_results.get(dwh_id) or {
            "dwhId": dwh_id, "companyName": None, "companyNameText": "", "descriptionText": "",
        }
        s3_row = s3_results.get(dwh_id) or {"dwhId": dwh_id, "s3Text": ""}
        existing_row = existing.get(dwh_id, {})
        scored = _score_company(es_row, s3_row, existing_row)
        baseline_fp = baseline.get(dwh_id)
        scored["baselineFP"] = baseline_fp
        scored["flippedFromBaseline"] = (
            baseline_fp is not None and bool(baseline_fp) != bool(scored["FP"])
        )
        output_rows.append(scored)
    return output_rows


def main(input_path: str, output_xlsx: str) -> None:
    if not ES_API_KEY:
        raise RuntimeError("ES_API_KEY is not set (check your .env / environment)")

    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    existing = _load_existing(dwh_ids)
    print(f"Loaded existing isAcquired/logo data for {len(existing)} dwhIds")

    baseline = _load_baseline(dwh_ids)
    print(f"Loaded baseline FP verdicts for {len(baseline)} dwhIds from {BASELINE_OUTPUT_XLSX}")

    data_source = ElasticCompanyDataSource(host=ES_HOST, index=ES_INDEX, api_key=ES_API_KEY)
    s3 = _s3_client()
    folders = list_pipeline_folders(s3, S3_BUCKET_NAME, S3_PREFIX)

    # Resume support: this machine has very little spare RAM (a full-batch
    # run gets killed by a system-wide low-memory guard partway through even
    # after chunking - see CHUNK_SIZE's comment), so a single process
    # finishing all 10k+ rows in one go isn't reliable here. Re-running this
    # same command picks up from the checkpoint CSV instead of starting
    # over - already-processed dwhIds are skipped.
    checkpoint_csv = output_xlsx.rsplit(".", 1)[0] + ".partial.csv"
    output_rows: list[dict[str, Any]] = []
    already_done: set[str] = set()
    if os.path.exists(checkpoint_csv):
        prior = pd.read_csv(checkpoint_csv, dtype={"dwhId": str})
        output_rows = prior.to_dict("records")
        already_done = set(prior["dwhId"])
        print(f"Resuming: {len(already_done)} dwhIds already checkpointed in {checkpoint_csv}")

    remaining = [d for d in dwh_ids if d not in already_done]
    print(f"\nProcessing {len(remaining)} remaining dwhIds in chunks of {CHUNK_SIZE} (ES + S3, live)...")
    print(f"(checkpointing progress to {checkpoint_csv} after every chunk)")
    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[start : start + CHUNK_SIZE]
        output_rows.extend(_process_chunk(chunk, existing, baseline, data_source, s3, folders))
        done = len(already_done) + min(start + CHUNK_SIZE, len(remaining))
        running_fp = sum(1 for r in output_rows if r["FP"])
        print(f"  {done}/{len(dwh_ids)}  (running FP count: {running_fp}/{len(output_rows)})")
        pd.DataFrame(output_rows).to_csv(checkpoint_csv, index=False)
        gc.collect()

    df_out = pd.DataFrame(
        output_rows,
        columns=[
            "dwhId", "companyName", "acquirerCompany", "recoveredAcquirer",
            "originalMatchedPattern", "confidenceScore", "matchedStages",
            "implausibleAcquirer", "knownBadPattern", "shortS3Acquirer",
            "esIsAcquired", "manuallyConfirmedAcquired", "FP", "baselineFP",
            "flippedFromBaseline",
        ],
    )
    df_out.to_excel(output_xlsx, index=False)

    fp_count = int(df_out["FP"].sum())
    print(f"\nWrote {len(df_out)} rows to {output_xlsx}")
    print(f"  false positives (FP=True): {fp_count} of {len(df_out)} ({fp_count / len(df_out):.1%})")
    print(f"\nFP breakdown (a row can trigger more than one check at once):")
    print(f"  FP via division_of pattern (check 7) : {int((df_out['FP'] & (df_out['originalMatchedPattern'] == 'division_of')).sum())}")
    print(f"  FP via implausible name (check 4)    : {int((df_out['FP'] & df_out['implausibleAcquirer']).sum())}")
    print(f"  FP via short S3-scrape acquirer (check 8): {int((df_out['FP'] & df_out['shortS3Acquirer']).sum())}")
    print(f"  (informational, no longer affects FP) knownBadPattern rows in this batch: {int(df_out['knownBadPattern'].sum())}")
    print(f"  (informational) esIsAcquired=True rows in this batch: {int(df_out['esIsAcquired'].sum())}"
          f" - all forced FP=False by the ES override (check 6), regardless of every other check")
    print(f"  (informational) recoveredAcquirer non-null (check 9 fired): {int(df_out['recoveredAcquirer'].notna().sum())}")
    print(f"  (informational) manuallyConfirmedAcquired rows: {int(df_out['manuallyConfirmedAcquired'].sum())}"
          f" - forced FP=False by explicit override, see _CONFIRMED_ACQUISITION_OVERRIDES")
    print(f"\nconfidenceScore distribution:")
    print(df_out["confidenceScore"].value_counts(dropna=False).sort_index())
    print(f"\nmatchedStages distribution (top 15):")
    print(df_out["matchedStages"].value_counts(dropna=False).head(15))
    if baseline:
        flipped = int(df_out["flippedFromBaseline"].sum())
        to_fp = int((df_out["flippedFromBaseline"] & df_out["FP"]).sum())
        to_not_fp = int((df_out["flippedFromBaseline"] & ~df_out["FP"]).sum())
        print(f"\nCompared to baseline ({BASELINE_OUTPUT_XLSX}):")
        print(f"  baseline FP count      : {sum(1 for v in baseline.values() if v)} of {len(baseline)}")
        print(f"  flipped verdict        : {flipped}")
        print(f"    not-FP -> FP         : {to_fp}")
        print(f"    FP -> not-FP         : {to_not_fp}")


if __name__ == "__main__":
    _input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_XLSX
    _output_xlsx = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_XLSX
    main(_input_path, _output_xlsx)
