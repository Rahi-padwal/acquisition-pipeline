"""Stage-3 acquisition detection: raw scraped website content stored in S3.

A THIRD, independent detection source alongside the existing companyName-
based (name_matching/check_name_signals.py) and description-based
(description_matching/description_detection_api.py) detectors. Those two
only see Elasticsearch's companyName/description fields; this one reads a
company's own scraped website prose (About pages, case studies, and -
notably - Terms of Service / Privacy Policy legal language identifying the
parent/contracting entity), which sometimes mentions an acquisition neither
of the other two sources ever see.

Bucket: gp-crustdata
Reads from 3 sibling prefixes under int-growthpal-crustdata/data_pipeline/v8/
(confirmed by direct bucket listing, not assumed):
  - milvus_expansion_scrape/ - dated pipeline_id=YYYY-MM-DD/ subfolders,
    most-recent wins on a re-scrape (see fetch_all_scrapes).
  - enrich_service_scrape/ - flat, one file per dwhId, no dating.
  - enrich_scrape/ - flat, one file per dwhId, no dating.
No priority ordering is assumed between the 3 sources (no evidence either
outranks another) - when a dwhId has data in more than one, ALL of it is
combined into one text blob rather than picking a winner (see
combined_text_blob_multi).
One JSON file per company, keyed by `{prefix}pipeline_id=YYYY-MM-DD/{dwhId}.json`
(confirmed against live bucket contents - the filename IS the dwhId, no
lookup table needed). The prefix currently holds 4 dated pipeline-run
subfolders; a company can appear in more than one (re-scraped) - the most
recent pipeline_id is treated as authoritative, and re-scrapes are logged
rather than silently resolved.

Bucket-label reliability: manual inspection of real samples found bucket
LABELS inside llm_context_buckets don't reliably match bucket CONTENTS
(e.g. "news" holding privacy/GDPR pages in one sample, brand pages in
another). Acquisition/ownership language has been observed specifically in
Terms of Service / Privacy Policy text, which lands in inconsistent
buckets - so this module concatenates every bucket's text EXCEPT the three
confirmed empty of acquisition-relevant content (pricing_structure,
products, integrations) rather than trusting bucket names to pick pages.

Standalone by design (own S3 client, own pattern set) - see
description_detection_api.py's docstring for why sibling detectors in this
project are kept standalone rather than sharing code, and note the same
reasoning: this reads different data, needs its own regex vocabulary, and
shouldn't require touching the ES-facing modules to change.

Run with: python s3_detection.py [input.csv] [db_path]
Input file: one dwhId per row, no header (same convention as
run_pipeline_on_file.py). Defaults to "test data.csv" / "test_combined_pipeline.db".

Deliberately NOT wired to run against the full S3 dataset - the input file
is always a specific, small dwhId list, and only those dwhIds' individual
keys are ever requested (no bucket-wide listing), so a small input file
naturally stays a small run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from sqlite3 import Connection, Row, connect
from typing import Any

import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parent.parent
for _subdir in ("name_matching", "description_matching"):
    _path = str(_ROOT / _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from check_name_signals import _company_identity_match  # noqa: E402 (from name_matching)
from check_name_signals import _PATTERN_BASE_CONFIDENCE as _NAME_PATTERN_BASE_CONFIDENCE  # noqa: E402
from description_detection_api import _PATTERN_BASE_CONFIDENCE as _DESCRIPTION_PATTERN_BASE_CONFIDENCE  # noqa: E402

from combined_check import (  # noqa: E402
    _CONFLICT_CAP,
    _CORROBORATION_BONUS,
    _CORROBORATION_CAP,
    _HIGH_TRUST_CONFLICT_CAP,
    _HIGH_TRUST_MIN_BASE_SCORE,
    classify_review_status,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_REGION = os.environ.get("S3_REGION")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
S3_PREFIX = os.environ.get("S3_PREFIX")

# Two sibling sources under the same v8/ root, confirmed by direct bucket
# listing (not assumed) to hold FLAT company JSON files (no pipeline_id=
# dated subfolders, unlike S3_PREFIX/milvus_expansion_scrape/) - one file
# per dwhId, same JSON shape as the milvus source. There's no dating signal
# here and no evidence either of these two, or milvus_expansion_scrape/,
# outranks the others - see combined_text_blob_multi() below for how that's
# handled (combine everything found rather than picking one as authoritative).
_V8_ROOT = S3_PREFIX.rsplit("milvus_expansion_scrape/", 1)[0] if S3_PREFIX else None
_ENRICH_SERVICE_PREFIX = (_V8_ROOT + "enrich_service_scrape/") if _V8_ROOT else None
_ENRICH_PREFIX = (_V8_ROOT + "enrich_scrape/") if _V8_ROOT else None

# Pure S3 HTTP calls (no ES round-trip involved) - safe to run at much
# higher concurrency than the ES-facing scripts' MAX_WORKERS=10. Bumped
# after a real production run showed stage 3's per-company S3 lookups (up
# to 6 calls each) dominating total runtime at the old worker count.
MAX_WORKERS = 64
SOURCE_NAME = "s3_scrape"

DEFAULT_INPUT_PATH = "test data.csv"
DEFAULT_DB_PATH = "test_combined_pipeline.db"
DEFAULT_OUTPUT_XLSX = "s3_detection_output.xlsx"

OUTPUT_COLUMNS = [
    "dwhId", "companyName", "isAcquired", "acquirerCompany", "source",
    "matchedPattern", "confidenceScore", "reviewStatus",
    "s3Outcome", "s3MatchedPattern", "s3Acquirer", "s3RawText", "s3FoundInPipelineIds",
]


# --- Pattern definitions ------------------------------------------------------
#
# A NEW, standalone pattern set for this source - deliberately not reusing
# description_detection_api's phrase list as-is, since scraped website prose
# (marketing copy, case studies, ToS/Privacy legal text) has different
# phrasing conventions than an ES `description` field. Same capture-group
# style as the rest of the project though: starts at a proper-noun-like
# token, non-greedy, stops at sentence-ending punctuation or end of string.
_NAME_CAPTURE = r"[A-Za-z0-9][^,.;:()\n]{0,49}?"
_NAME_TERMINATOR = r"(?=[,.;:)]|$)"

_WAS_ACQUIRED_BY = re.compile(
    r"was\s+acquired\s+by\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
_WHOLLY_OWNED_SUBSIDIARY = re.compile(
    r"(?:operates\s+as\s+an?\s+)?wholly[\s-]+owned\s+subsidiary\s+of\s+(?P<acquirer>" + _NAME_CAPTURE + r")"
    + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_IS_A_SUBSIDIARY_OF = re.compile(
    r"\bis\s+an?\s+subsidiary\s+of\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
# Removed: "part of X" used to live here, scored 1.0. Pulled entirely after
# testing against 300 real S3-scraped companies found it overwhelmingly
# false-positive in raw website prose - "part of" is extremely common
# generic English phrasing outside any ownership context ("part of the fee",
# "part of my daily habit now", "part of bankruptcy", "part of aging"). This
# same pattern is reliable at 1.0 in name_matching/check_name_signals.py and
# description_detection_api.py because there it's matched against a
# structured companyName field or a formal ES description sentence, neither
# of which carries this source's volume of unrelated marketing copy - the
# scale doesn't transfer. 52 of 59 matches in the 300-company test batch
# were this pattern, and nearly all were garbage; do not re-add without a
# guard that filters out non-proper-noun captures.

_DIVISION_OF = re.compile(
    r"\ba\s+division\s+of\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
_OPERATING_UNIT_OF = re.compile(
    r"\ban?\s+operating\s+unit\s+of\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
# Acquirer-first, self-second: "[X] acquired [Company]". Both names are
# stated explicitly here, so - unlike most patterns above - the match is
# verified against the SUBJECT company's own name (see
# _mentions_subject_company) before being accepted. Without that guard this
# pattern would fire on ANY "X acquired Y" sentence in the blob, including
# a case study describing one of the subject's CLIENTS being acquired by
# someone else - the exact misattribution risk flagged for this whole
# source (case-study/testimonial text describes a client's acquisition, not
# the subject's own).
#
# (?!by\b) is required: without it, this pattern also matches PASSIVE-voice
# sentences ("CompXL acquired by Salary.com", "GoSpotCheck Acquired by
# FORM" - both real taglines found in testing) and gets the roles
# backwards, since the acquirer capture grabs whatever precedes the literal
# word "acquired" regardless of voice - "CompXL acquired by Salary.com"
# was misread as acquirer="CompXL" (the SUBJECT, acquired FROM Salary.com,
# not the other way around). The same gap let a Privacy Policy's routine
# business-transfer boilerplate ("if we or our affiliates are or may be
# acquired by...", present on nearly every company's ToS regardless of
# whether an acquisition ever happened) fire as a false match too, since
# its target capture also started right after "acquired by". Excluding
# "acquired by" here stops both: genuine passive-voice acquisitions are
# still caught correctly by was_acquired_by above (which requires the
# leading "was" and stays a distinct, correctly-directed pattern).
#
# Optional (?:has\s+|have\s+|had\s+)? before "acquired": without it, present-
# perfect phrasing ("Sprockets has Acquired HourWork" - a real match found
# in testing) gets the auxiliary verb swallowed into the acquirer capture
# ("Sprockets has") instead of stopping at the real name, since the capture
# is non-greedy but still has to reach the literal "acquired" token somehow.
_ACQUIRER_ACQUIRES_SELF = re.compile(
    r"(?P<acquirer>" + _NAME_CAPTURE + r")\s+(?:has\s+|have\s+|had\s+)?acquired\s+(?!by\b)"
    r"(?P<target>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR,
    re.IGNORECASE,
)
_JOINED_FORCES = re.compile(
    r"has\s+joined\s+forces\s+with\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)

# Legal-document-specific (Terms of Service / Privacy Policy phrasing).
_PROVIDED_BY = re.compile(
    r"this\s+Service\s+is\s+provided\s+by\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
# Removed: "[Company] is owned by X" used to live here, scored 1.0. Pulled
# after the same 300-company test batch caught it matching ToS/Privacy
# boilerplate about CONTENT ownership rather than corporate parent
# ownership - e.g. "T Bar Inc is owned by us and our licensors" (the "us"
# there is the site's own operator describing copyright, not a parent
# company). Do not re-add without a guard rejecting boilerplate acquirer
# text ("us", "we", "its licensors", "its affiliates", etc.).

# Backward capture: acquirer appears BEFORE the trigger phrase ("Acme
# Holdings, its parent company"). Relies on search()'s leftmost-match
# behavior to pull in the whole preceding name run up to the comma - see
# module history/PR notes if this needs adjusting, non-greedy quantifiers
# at a fixed starting position still expand until the rest of the pattern
# can match, so this naturally captures the full preceding phrase.
_PARENT_COMPANY = re.compile(
    r"(?P<acquirer>[A-Za-z0-9][^,.;:()\n]{0,49}?),\s+its\s+parent\s+company\b", re.IGNORECASE
)
_COMPANY_WITHIN_GROUP = re.compile(
    r"a\s+company\s+within\s+the\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+group", re.IGNORECASE
)

# (matched_pattern, compiled pattern, base confidence). Priority order -
# first pattern (checked in this order) to produce an accepted match wins;
# later patterns in the list are never tried once one hits. Matches the
# scale established earlier in this project: explicit, unambiguous
# ownership/acquisition language scores 1.0, "joined forces" stays at 0.7
# given its partnership-announcement ambiguity (see
# _is_joined_forces_list_announcement below).
_S3_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("was_acquired_by", _WAS_ACQUIRED_BY, 1.0),
    ("wholly_owned_subsidiary", _WHOLLY_OWNED_SUBSIDIARY, 1.0),
    ("is_a_subsidiary_of", _IS_A_SUBSIDIARY_OF, 1.0),
    ("division_of", _DIVISION_OF, 1.0),
    ("operating_unit_of", _OPERATING_UNIT_OF, 1.0),
    ("acquirer_acquires_self", _ACQUIRER_ACQUIRES_SELF, 1.0),
    ("joined_forces_with", _JOINED_FORCES, 0.7),
    ("provided_by", _PROVIDED_BY, 1.0),
    ("parent_company", _PARENT_COMPANY, 1.0),
    ("company_within_group", _COMPANY_WITHIN_GROUP, 1.0),
]

_PATTERN_BASE_CONFIDENCE: dict[str, float] = {name: score for name, _, score in _S3_PATTERNS}


def _clean_acquirer(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw).strip().strip(")").strip()
    # Backward captures (parent_company) and generic prefixes ("part of THE
    # Acme Group") can pick up a leading article - strip it.
    return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()


# --- General garbage-capture guard, applied to every pattern's final
# acquirer string ---------------------------------------------------------
#
# NAME_CAPTURE only stops at a small set of punctuation marks (see its
# definition above) - when the real acquirer name isn't reachable (internal
# commas/parens the capture can't cross, e.g. "Total Resource Management,
# Inc. (TRM)") or a sentence boundary has no punctuation at all (two
# headings concatenated back-to-back in scraped markdown), the capture
# either falls back to a nearby filler word or balloons across an entire
# extra clause. Two real examples from testing:
#   - "has acquired Stratum Consulting Partners" -> acquirer captured as
#     "has" (the true acquirer's name was unreachable behind "(TRM),").
#   - "TTC Group Acquires Miad Healthcare TTC Group has acquired Miad
#     Healthcare" -> acquirer captured as the entire first clause, because
#     nothing punctuation-wise separated the two back-to-back mentions.
# Rather than trying to solve unbounded-prose segmentation, reject captures
# that show clear signs of failure instead: no proper-noun-like word at
# all, the word "acquir-" showing up INSIDE the capture (a strong tell that
# the boundary crossed into a second acquisition-language clause), or an
# implausible word count for a company name.
_GARBAGE_ACQUIRER_RE = re.compile(r"\bacquir", re.IGNORECASE)
_MAX_ACQUIRER_WORDS = 6
# Blog-headline phrasing ("Why We Acquired HourWork") capitalizes pronouns
# and question words the same way a real company name would be capitalized,
# so the "has at least one capitalized word" check alone lets them through -
# reject a capture that's ENTIRELY made of these, even if every word in it
# happens to be capitalized.
_NON_COMPANY_WORDS = {
    "we", "why", "us", "our", "ours", "they", "them", "their", "it", "its",
    "this", "that", "these", "those", "who", "how", "what", "when",
}


def _is_garbage_acquirer(acquirer: str) -> bool:
    if _GARBAGE_ACQUIRER_RE.search(acquirer):
        return True
    words = acquirer.split()
    if len(words) > _MAX_ACQUIRER_WORDS:
        return True
    if not any(w[:1].isupper() for w in words):
        return True
    if all(w.strip(".,'-&").lower() in _NON_COMPANY_WORDS for w in words):
        return True
    # Bare legal suffix with nothing else ("LLC", "Inc." - a real conflict
    # found in testing) - strip suffixes/punctuation and reject if nothing
    # proper-noun-like is left.
    stripped = _LEGAL_SUFFIX_RE.sub(" ", acquirer).strip(" .,")
    return not stripped


# --- acquirer_acquires_self guard: does the captured "target" actually name
# the subject company, not a client mentioned in a case study? ---------------

_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings|gmbh|plc|srl|sa|nv|bv)\b\.?", re.IGNORECASE
)


def _normalize_company_name(name: str) -> str:
    name = _LEGAL_SUFFIX_RE.sub(" ", name or "")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _mentions_subject_company(candidate_text: str, company_name: str | None) -> bool:
    norm_subject = _normalize_company_name(company_name or "")
    if len(norm_subject) < 3:
        return False
    norm_candidate = _normalize_company_name(candidate_text)
    return norm_subject in norm_candidate or norm_candidate in norm_subject


def _is_self_reference(acquirer_text: str, company_name: str | None) -> bool:
    """Reject a captured "acquirer" that's just the subject naming itself.
    Two real bugs caught in testing motivate the asymmetric check below
    (exact-equality alone isn't enough):

    1. "FCamara" matching company_within_group on "a company within the
       FCamara group" (the page describing itself) - an exact match, caught
       by equality alone.
    2. A company's own "About Us" page describing ITS OWN subsidiary -
       "wca Environment (IRELAND) limited... is a subsidiary of wca
       environment limited [registered in Ireland]" - the subject
       ("wca environment limited") is the PARENT here, not the acquired
       party, but the acquirer capture picked up the subject's name plus
       trailing prose ("registered in Ireland") and exact equality missed
       it entirely.

    So: reject whenever the captured acquirer text STARTS WITH the
    subject's full normalized name (case 2 - subject name plus trailing
    junk). Do NOT reject the reverse direction - a genuine subsidiary is
    very often named after its parent with extra words appended ("MIESCOR
    Telecommunications Business Unit" IS a real subsidiary of "MIESCOR",
    where the SUBJECT's own name, not the acquirer capture, is the longer
    one) - that's exactly the corroborating case this guard must NOT
    reject, and the startswith direction below correctly leaves it alone."""
    norm_subject = _normalize_company_name(company_name or "")
    if len(norm_subject) < 3:
        return False
    return _normalize_company_name(acquirer_text).startswith(norm_subject)


# --- joined_forces_with list-announcement guard ------------------------------
#
# Same ambiguity risk already handled for the ES description path's
# joined_forces pattern (description_detection_api.py): "has joined forces
# with X" can announce a genuine acquisition OR a partnership/client-roster
# ("has joined forces with industry giants like X", "...with X, Y, and Z").
# Duplicated here rather than imported - see this module's docstring on
# staying standalone.
_JOINED_FORCES_LIST_CONNECTOR = re.compile(
    r"(?:like|such\s+as|including|comprising|e\.g\.?|to\s+name\s+(?:a\s+few|just\s+one|but\s+a\s+few))\s*$",
    re.IGNORECASE,
)
_JOINED_FORCES_MULTI_ENTITY_TAIL = re.compile(r"^(?:,\s*[A-Z][\w&.'-]*|\s+and\s+[A-Z])")
_JOINED_FORCES_MULTI_ENTITY_INTERNAL_AND = re.compile(r"[\w&.'-]+\s+and\s+[A-Z]")


def _is_joined_forces_list_announcement(text: str, match: re.Match) -> bool:
    lead = text[max(0, match.start("acquirer") - 60) : match.start("acquirer")]
    if _JOINED_FORCES_LIST_CONNECTOR.search(lead):
        return True
    if _JOINED_FORCES_MULTI_ENTITY_INTERNAL_AND.search(match.group("acquirer")):
        return True
    tail = text[match.end("acquirer") : match.end("acquirer") + 60]
    return bool(_JOINED_FORCES_MULTI_ENTITY_TAIL.match(tail))


# --- General fix for the whole "which company is this sentence even about?"
# class of false positive (generalizes past the single wca-environment bug
# that motivated it) ----------------------------------------------------
#
# was_acquired_by / wholly_owned_subsidiary / is_a_subsidiary_of /
# division_of / operating_unit_of / joined_forces_with are all
# grammatically "[SELF] <trigger phrase> [ACQUIRER]" - the trigger
# phrase's SUBJECT is whichever company the sentence is actually describing,
# which is NOT always the dwhId's own subject company. Real example caught
# in testing: a company's own "About Us" page describing ITS OWN subsidiary
# ("wca Environment (IRELAND) limited... is a subsidiary of wca environment
# limited") - the subject there is the PARENT, not the acquired party, and
# a different, unrelated entity ("wca Environment (IRELAND) limited") is
# the one actually being described as "a subsidiary of". The same
# structural gap is exactly the case-study-misattribution risk documented
# below (a page mentioning a CLIENT's acquisition, not the subject's own).
#
# Rather than special-casing "the acquirer capture happens to equal the
# subject" (which is what _is_self_reference alone catches), this verifies
# the actual grammatical subject directly - the same principle
# acquirer_acquires_self already applies to its own target via
# _mentions_subject_company, generalized to every pattern with this shape.
# provided_by, parent_company, and company_within_group are deliberately
# excluded: provided_by's ToS phrasing rarely has the company name in
# prose immediately before it (a strict check would mostly just reject
# genuine matches); parent_company's self entity is typically referenced
# by pronoun ("we/our") rather than by name; and company_within_group's
# natural real-world phrasing is self-referential without ever restating
# the subject's own name ("We are a company within the X group" - a real
# case tested and confirmed to false-reject under this check), unlike the
# patterns below where every confirmed real match this session explicitly
# named the subject as the sentence's grammatical subject.
_SUBJECT_PROXIMITY_WINDOW = 150
_PATTERNS_REQUIRING_NEARBY_SUBJECT = {
    "was_acquired_by", "wholly_owned_subsidiary", "is_a_subsidiary_of",
    "division_of", "operating_unit_of", "joined_forces_with",
}


def _subject_mentioned_nearby(text: str, match_start: int, company_name: str | None) -> bool:
    """Reuses _mentions_subject_company's lenient bidirectional substring
    check (not a stricter one) - a strict "window must contain the FULL
    company_name" check breaks a common real shape where companyName
    itself carries an acquirer suffix ("Mutual Mobile | A Grid Dynamics
    Company") that the page's own prose never repeats (it just says
    "Mutual Mobile has joined forces with..."). The wca-environment-style
    bug this guard targets is still independently caught by
    _is_self_reference below (applied to the cleaned acquirer capture,
    not this window) even on the rare case this lenient check passes
    something it shouldn't - see that function."""
    if len(_normalize_company_name(company_name or "")) < 3:
        return True  # nothing to check the window against - don't block on it
    window = text[max(0, match_start - _SUBJECT_PROXIMITY_WINDOW) : match_start]
    return _mentions_subject_company(window, company_name)


def extract_acquirer_from_s3_text(text: str, company_name: str | None = None) -> dict[str, Any] | None:
    """Scan `text` (the combined bucket blob for one company) for
    acquisition/ownership language. Returns None if nothing matches,
    otherwise {"acquirerCompany", "matchedPattern", "confidenceScore", "rawText"}.

    Known false-positive risks:
      1. Case-study/testimonial text describing a CLIENT's acquisition (or
         a company's own unrelated subsidiary), not the subject's own -
         guarded for was_acquired_by, wholly_owned_subsidiary,
         is_a_subsidiary_of, division_of, operating_unit_of,
         joined_forces_with, and company_within_group via
         _subject_mentioned_nearby (requires the subject's name to appear
         shortly before the trigger phrase - see that function), and for
         acquirer_acquires_self via _mentions_subject_company on its
         target. NOT guarded for provided_by or parent_company - see
         _PATTERNS_REQUIRING_NEARBY_SUBJECT's comment for why.
      2. joined_forces_with can still be a partnership announcement even
         after the list-announcement guard - that guard catches the common
         multi-name/list-connector shapes, not every ambiguous phrasing.
    """
    if not text:
        return None

    for pattern_name, pattern, base_score in _S3_PATTERNS:
        m = pattern.search(text)
        if m is None:
            continue

        if pattern_name == "acquirer_acquires_self" and not _mentions_subject_company(
            m.group("target"), company_name
        ):
            continue
        if pattern_name == "joined_forces_with" and _is_joined_forces_list_announcement(text, m):
            continue
        if pattern_name in _PATTERNS_REQUIRING_NEARBY_SUBJECT and not _subject_mentioned_nearby(
            text, m.start(), company_name
        ):
            continue

        acquirer = _clean_acquirer(m.group("acquirer"))
        if not acquirer or _is_garbage_acquirer(acquirer):
            continue
        # A company cannot be acquired by / a subsidiary of / a division of
        # itself - reject any capture that just names the subject back
        # (real example caught in testing: "FCamara" matching
        # company_within_group on "a company within the FCamara group",
        # where the page was describing itself, not a parent).
        if pattern_name != "acquirer_acquires_self" and _is_self_reference(acquirer, company_name):
            continue

        return {
            "acquirerCompany": acquirer,
            "matchedPattern": pattern_name,
            "confidenceScore": base_score,
            "rawText": m.group(0)[:300],
        }

    return None


# --- Text-blob extraction from the S3 JSON shape -----------------------------
#
# CONFIRMED (manual inspection of two real samples): bucket LABELS inside
# llm_context_buckets don't reliably match bucket CONTENTS - "news" held
# brand pages in one sample and privacy/GDPR legal pages in another;
# "solution" held a terms.php page in another. Acquisition/ownership
# language has specifically been observed in Terms of Service / Privacy
# Policy pages, which land inconsistently across buckets. So: include every
# bucket's text EXCEPT the three confirmed empty of acquisition-relevant
# content, rather than cherry-picking by bucket name.
_EXCLUDED_BUCKETS = {"pricing_structure", "products", "integrations"}


def combined_text_blob(scrape_json: dict[str, Any]) -> str:
    buckets = ((scrape_json.get("scrape") or {}).get("llm_context_buckets")) or {}
    parts: list[str] = []
    for bucket_name, entries in buckets.items():
        if bucket_name in _EXCLUDED_BUCKETS:
            continue
        for entry in entries or []:
            text = (entry or {}).get("text")
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def combined_text_blob_multi(scrape_jsons: list[dict[str, Any]]) -> str:
    """Same bucket-inclusion rule as combined_text_blob, applied across
    MULTIPLE scrape JSONs for the same company (milvus_expansion_scrape/ +
    enrich_service_scrape/ + enrich_scrape/, whichever were found) and
    joined into one blob. Everything found gets combined rather than one
    source being picked as authoritative - see the module-level prefix
    constants for why (no dating signal on the two flat sources, no
    evidence either outranks milvus)."""
    return "\n\n".join(blob for j in scrape_jsons if (blob := combined_text_blob(j)))


# --- S3 access ----------------------------------------------------------------

_PIPELINE_FOLDER_RE = re.compile(r"pipeline_id=(\d{4}-\d{2}-\d{2})/$")


def _s3_client():
    # max_pool_connections must cover MAX_WORKERS concurrent requests -
    # botocore's default urllib3 pool (10) would otherwise silently
    # serialize requests behind that cap regardless of thread count.
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=S3_REGION,
        config=Config(max_pool_connections=max(MAX_WORKERS, 10) + 10),
    )


def list_pipeline_folders(s3, bucket: str, prefix: str) -> list[str]:
    """All `pipeline_id=YYYY-MM-DD/` subfolders under prefix, newest first -
    discovered from the live bucket rather than hardcoded, so a new dated
    run folder is picked up automatically."""
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    dated = []
    for cp in resp.get("CommonPrefixes", []):
        folder = cp["Prefix"]
        m = _PIPELINE_FOLDER_RE.search(folder)
        if m:
            dated.append((m.group(1), folder))
    dated.sort(key=lambda pair: pair[0], reverse=True)
    return [folder for _, folder in dated]


def _fetch_company_json(s3, bucket: str, folder: str, dwh_id: str) -> dict[str, Any] | None:
    key = f"{folder}{dwh_id}.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(obj["Body"].read())


def fetch_latest_scrape(s3, bucket: str, folders: list[str], dwh_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Fetch dwh_id's JSON from every folder it exists in (folders must be
    newest-first). Returns (json_from_the_newest_folder_it_was_found_in,
    [every folder it was found in]) - `data` is None if not found anywhere.

    Re-scrapes (found in more than one folder) are resolved by picking the
    most recent, not silently - callers should log `found_in` when it has
    more than one entry."""
    found_in: list[str] = []
    data: dict[str, Any] | None = None
    for folder in folders:
        candidate = _fetch_company_json(s3, bucket, folder, dwh_id)
        if candidate is not None:
            found_in.append(folder)
            if data is None:
                data = candidate  # folders is newest-first, so the first hit wins
    return data, found_in


def fetch_all_scrapes(
    s3, bucket: str, milvus_folders: list[str], dwh_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch dwh_id's JSON from all 3 top-level S3 sources: the dated
    milvus_expansion_scrape/ folders (most-recent wins, re-scrapes logged -
    see fetch_latest_scrape) plus the two flat sources enrich_service_scrape/
    and enrich_scrape/ (at most one file each, no dating). Returns
    (list_of_json_found, list_of_source_labels_they_came_from) - both may be
    empty if dwh_id was never scraped anywhere."""
    found: list[dict[str, Any]] = []
    sources: list[str] = []

    milvus_data, milvus_found_in = fetch_latest_scrape(s3, bucket, milvus_folders, dwh_id)
    if milvus_data is not None:
        found.append(milvus_data)
        sources.append(milvus_found_in[0])
        if len(milvus_found_in) > 1:
            logger.info(
                "dwhId %s: re-scraped within milvus_expansion_scrape/, found in %s - using %s as authoritative",
                dwh_id, milvus_found_in, milvus_found_in[0],
            )

    for label, prefix in (("enrich_service_scrape", _ENRICH_SERVICE_PREFIX), ("enrich_scrape", _ENRICH_PREFIX)):
        if prefix is None:
            continue
        data = _fetch_company_json(s3, bucket, prefix, dwh_id)
        if data is not None:
            found.append(data)
            sources.append(label)

    return found, sources


# --- Cross-source corroboration/conflict, applied against an EXISTING row ----
#
# This duplicates combined_check.py's _apply_cross_source_modifiers /
# _conflict_capped_score math rather than importing them directly, because
# those functions decide "high-trust" by checking a matched_pattern's base
# score in ONLY the name_matching and description_detection_api tables -
# several of this module's own pattern names (provided_by, is_owned_by,
# parent_company, company_within_group) don't exist in either table and
# would silently fall through to the LOW-trust conflict cap even though
# they're scored 1.0 here. This version checks all three pattern tables.
def _is_high_trust_pattern(matched_pattern: str) -> bool:
    base = (
        _NAME_PATTERN_BASE_CONFIDENCE.get(matched_pattern)
        or _DESCRIPTION_PATTERN_BASE_CONFIDENCE.get(matched_pattern)
        or _PATTERN_BASE_CONFIDENCE.get(matched_pattern)
    )
    return base is not None and base >= _HIGH_TRUST_MIN_BASE_SCORE


def _conflict_capped_score(score: float, matched_pattern: str) -> float:
    cap = _HIGH_TRUST_CONFLICT_CAP if _is_high_trust_pattern(matched_pattern) else _CONFLICT_CAP
    return min(score, cap)


def _apply_cross_source_modifiers(
    primary_acquirer: str, primary_score: float, other_acquirer: str, primary_pattern: str
) -> tuple[float, str]:
    if _company_identity_match(primary_acquirer, other_acquirer):
        return min(_CORROBORATION_CAP, primary_score + _CORROBORATION_BONUS), "corroborated_by_both_sources"
    return _conflict_capped_score(primary_score, primary_pattern), "source_conflict"


# --- DB merge -------------------------------------------------------------


def _create_table_if_missing(conn: Connection) -> None:
    """Same schema test_combined_pipeline.py/run_full_batch.py already use -
    this module never creates a new table, only adds rows/updates existing
    ones (per the task constraint: additive, no new table)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS combined_acquisitions (
            dwhId TEXT PRIMARY KEY,
            companyName TEXT,
            isAcquired INTEGER,
            acquirerCompany TEXT,
            source TEXT,
            matchedPattern TEXT,
            confidenceScore REAL,
            reviewStatus TEXT
        )
        """
    )
    conn.commit()


def merge_s3_result_into_db(
    conn: Connection, dwh_id: str, company_name: str | None, s3_result: dict[str, Any] | None
) -> str:
    """Write `s3_result` (or lack of one) for `dwh_id` into combined_acquisitions,
    honoring whatever row is already there. Returns a short outcome label for
    the run summary: "no_match" | "fresh_insert" | "upgraded_from_empty" |
    "corroborated" | "conflict_capped".

    Never silently overwrites an existing acquirer - see the branches below.
    """
    conn.row_factory = Row
    existing = conn.execute("SELECT * FROM combined_acquisitions WHERE dwhId = ?", (dwh_id,)).fetchone()

    if s3_result is None:
        # No S3 evidence either way - never downgrade or fabricate a
        # "not acquired" verdict over whatever's already there. Only write a
        # fresh not-acquired row when this dwhId has never been seen before.
        if existing is None:
            is_acquired, review_status = classify_review_status(None)
            conn.execute(
                "INSERT INTO combined_acquisitions "
                "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
                "VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?)",
                (dwh_id, company_name, is_acquired, review_status),
            )
        return "no_match"

    acquirer, pattern, score = s3_result["acquirerCompany"], s3_result["matchedPattern"], s3_result["confidenceScore"]

    if existing is None or not existing["acquirerCompany"]:
        # Fresh detection: either dwhId never seen before, or it was seen
        # but no other source ever found an acquirer for it (nothing to
        # corroborate/conflict against) - the S3 evidence stands on its own.
        is_acquired, review_status = classify_review_status(score)
        conn.execute(
            "INSERT OR REPLACE INTO combined_acquisitions "
            "(dwhId, companyName, isAcquired, acquirerCompany, source, matchedPattern, confidenceScore, reviewStatus) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (dwh_id, company_name, is_acquired, acquirer, SOURCE_NAME, pattern, score, review_status),
        )
        return "fresh_insert" if existing is None else "upgraded_from_empty"

    # Existing row already has an acquirer from another source - apply the
    # same corroboration/conflict logic combined_check.py uses, keyed off
    # the EXISTING (already-established) pattern's trust level. The existing
    # acquirer/source/matchedPattern are kept either way (matches the
    # convention in test_combined_pipeline.py's own conflict branch: the
    # higher-confidence/primary side's identity wins, only the score moves) -
    # only confidenceScore/isAcquired/reviewStatus are updated.
    adjusted_score, notes = _apply_cross_source_modifiers(
        existing["acquirerCompany"], existing["confidenceScore"] or 0.0, acquirer, existing["matchedPattern"]
    )
    is_acquired, review_status = classify_review_status(adjusted_score)
    conn.execute(
        "UPDATE combined_acquisitions SET isAcquired = ?, confidenceScore = ?, reviewStatus = ? WHERE dwhId = ?",
        (is_acquired, adjusted_score, review_status, dwh_id),
    )

    if notes == "source_conflict":
        logger.warning(
            "dwhId %s: S3 conflict - existing acquirer %r (%s) vs S3 acquirer %r (%s) - "
            "keeping existing, score capped to %s",
            dwh_id, existing["acquirerCompany"], existing["matchedPattern"], acquirer, pattern, adjusted_score,
        )
        return "conflict_capped"

    logger.info(
        "dwhId %s: S3 corroborates existing acquirer %r - score %s -> %s",
        dwh_id, existing["acquirerCompany"], existing["confidenceScore"], adjusted_score,
    )
    return "corroborated"


# --- Driver --------------------------------------------------------------


def _read_dwh_ids(path: str) -> list[str]:
    read = pd.read_excel if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv
    df = read(path, header=None, dtype=str)
    return [v.strip() for v in df.iloc[:, 0].dropna().tolist() if v.strip()]


def process_dwh_id(s3, bucket: str, folders: list[str], dwh_id: str) -> dict[str, Any]:
    """S3 fetch + extraction only - no DB access (DB writes happen single-
    threaded afterward, same discipline as test_combined_pipeline.py)."""
    scrapes, found_in = fetch_all_scrapes(s3, bucket, folders, dwh_id)
    if not scrapes:
        return {"dwhId": dwh_id, "companyName": None, "found_in": [], "s3_result": None, "no_data": True}

    company_name = next((s.get("companyName") for s in scrapes if s.get("companyName")), None)
    blob = combined_text_blob_multi(scrapes)
    s3_result = extract_acquirer_from_s3_text(blob, company_name)
    return {"dwhId": dwh_id, "companyName": company_name, "found_in": found_in, "s3_result": s3_result, "no_data": False}


def main(input_path: str, db_path: str, output_xlsx: str) -> None:
    if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME and S3_PREFIX):
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_BUCKET_NAME / S3_PREFIX must all be set (check .env)"
        )

    dwh_ids = _read_dwh_ids(input_path)
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    s3 = _s3_client()
    folders = list_pipeline_folders(s3, S3_BUCKET_NAME, S3_PREFIX)
    print(f"Pipeline folders (newest first): {folders}")
    if not folders:
        raise RuntimeError(f"No pipeline_id=... folders found under s3://{S3_BUCKET_NAME}/{S3_PREFIX}")

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_dwh_id, s3, S3_BUCKET_NAME, folders, dwh_id): dwh_id for dwh_id in dwh_ids
        }
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  fetched {done}/{len(dwh_ids)}")

    conn = connect(db_path)
    _create_table_if_missing(conn)

    outcomes: dict[str, int] = {}
    no_data = 0
    matched_samples: list[dict[str, Any]] = []
    for row in results:
        if row["no_data"]:
            no_data += 1
            row["outcome"] = "no_s3_data"
            continue
        outcome = merge_s3_result_into_db(conn, row["dwhId"], row["companyName"], row["s3_result"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        row["outcome"] = outcome
        if row["s3_result"] is not None:
            matched_samples.append(row)

    # Pull the FINAL post-merge state for every dwhId touched (whether this
    # run wrote it or it already existed) - this is what actually landed in
    # the DB after corroboration/conflict logic ran, not just this run's raw
    # S3 extraction, so the Excel report reflects the real verdict.
    conn.row_factory = Row
    placeholders = ",".join("?" * len(dwh_ids))
    db_rows = {
        r["dwhId"]: dict(r)
        for r in conn.execute(
            f"SELECT * FROM combined_acquisitions WHERE dwhId IN ({placeholders})", dwh_ids
        ).fetchall()
    }
    conn.commit()
    conn.close()

    print(f"\nProcessed {len(dwh_ids)} dwhIds against s3://{S3_BUCKET_NAME}/{S3_PREFIX}")
    print(f"  no_s3_data (not scraped)   = {no_data}")
    for label in ("no_match", "fresh_insert", "upgraded_from_empty", "corroborated", "conflict_capped"):
        print(f"  {label:<24} = {outcomes.get(label, 0)}")

    print(f"\nSample of matched rows for manual verification (up to 20 of {len(matched_samples)}):")
    for row in matched_samples[:20]:
        r = row["s3_result"]
        print(
            f"  dwhId={row['dwhId']} companyName={row['companyName']!r} outcome={row['outcome']} "
            f"pattern={r['matchedPattern']} score={r['confidenceScore']} acquirer={r['acquirerCompany']!r} "
            f"rawText={r['rawText']!r}"
        )

    output_rows = []
    for row in results:
        db_row = db_rows.get(row["dwhId"], {})
        s3_result = row["s3_result"]
        output_rows.append(
            {
                "dwhId": row["dwhId"],
                "companyName": db_row.get("companyName", row["companyName"]),
                "isAcquired": db_row.get("isAcquired"),
                "acquirerCompany": db_row.get("acquirerCompany"),
                "source": db_row.get("source"),
                "matchedPattern": db_row.get("matchedPattern"),
                "confidenceScore": db_row.get("confidenceScore"),
                "reviewStatus": db_row.get("reviewStatus"),
                "s3Outcome": row["outcome"],
                "s3MatchedPattern": s3_result["matchedPattern"] if s3_result else None,
                "s3Acquirer": s3_result["acquirerCompany"] if s3_result else None,
                "s3RawText": s3_result["rawText"] if s3_result else None,
                "s3FoundInPipelineIds": ", ".join(row["found_in"]) if row["found_in"] else None,
            }
        )

    order = {dwh_id: i for i, dwh_id in enumerate(dwh_ids)}
    output_rows.sort(key=lambda r: order.get(r["dwhId"], len(dwh_ids)))
    df_out = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    df_out.to_excel(output_xlsx, index=False)
    print(f"\nWrote {len(df_out)} rows to {output_xlsx}")


if __name__ == "__main__":
    _input_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_PATH
    _db_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DB_PATH
    _output_xlsx = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_OUTPUT_XLSX
    main(_input_path, _db_path, _output_xlsx)
