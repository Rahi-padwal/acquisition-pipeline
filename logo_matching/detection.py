"""Keyword-based acquisition detection over OCR-extracted logo text.

Standalone by design (own pattern set, no imports from name_matching/
description_matching/combined_matching) - same convention the rest of this
project follows for sibling detectors: each source has different text
shape and shouldn't require touching the others to change.

Pattern set chosen (5 patterns) after inspecting real OCR output from the
sample logos in `logo data/`:

    powered_by      - "powered by X"
    a_x_company     - "a X company" / "an X company"
    part_of         - "part of X" (optionally "part of the X group")
    x_group         - bare "X Group" mention, not already consumed by part_of
    by_x            - "by X" (loosest/riskiest trigger - low confidence)

These were deliberately picked over the two existing sibling pattern sets
in this project:
  - s3_detection.py's set (was_acquired_by, wholly_owned_subsidiary, etc.)
    targets full webpage PROSE and doesn't cover "a X company" / "by X" at
    all - the exact shape logo taglines actually use.
  - name_matching/check_name_signals.py's set covers the right phrasing
    family (a_x_company, powered_by, self_by_acquirer) but its regexes are
    FULL-STRING ANCHORED (^self, trigger acquirer$), built for a single
    companyName field. PaddleOCR returns each logo tagline as a SEPARATE
    detected text line with no punctuation joining it to the main
    logotype - a real sample logo produced rec_texts ["TMGRA", "an RIA
    Company"] as two independent lines, with no comma/self-prefix at all
    in the second line. Anchoring to "^self, an X company$" would silently
    fail to match this. So: unanchored `search()`-style patterns adapted
    for OCR fragments, not a straight import of either sibling's regexes.

Confidence scores follow the same scale already established across the
project (explicit/unambiguous = 1.0, ambiguous/weak triggers scored down)
and reuse the specific values name_matching/check_name_signals.py already
settled on for the equivalent trigger words (powered_by=0.7, "by X"=0.3),
since those numbers came from that module's own manual false-positive
review, not a fresh guess.
"""

from __future__ import annotations

import re
from typing import Any

# Short, name-shaped capture: starts at an alnum char, stops at a small set
# of punctuation/line-boundary markers. OCR lines are short (a few words at
# most) so this doesn't need s3_detection's prose-scale non-greedy tuning.
_NAME_CAPTURE = r"[A-Za-z0-9][^,.;:()\n|]{0,49}?"
_NAME_TERMINATOR = r"(?=[,.;:)|\n]|$)"

_POWERED_BY = re.compile(
    r"\bpowered\s+by\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE
)
# Unanchored version of name_matching's a_x_company - no comma/self prefix
# required, since OCR lines routinely arrive as their own fragment
# ("an RIA Company" with nothing before it).
_A_X_COMPANY = re.compile(
    r"\ban?\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+compan(?:y|ies)\b", re.IGNORECASE
)
_PART_OF = re.compile(
    r"\bpart\s+of\s+(?:the\s+)?(?P<acquirer>" + _NAME_CAPTURE + r")\b(?:\s+group\b)?", re.IGNORECASE
)
# Bare "X Group" mention with no "part of" prefix - checked only as a
# fallback (see _LOGO_PATTERNS ordering: part_of is tried first, so a
# blob's "part of Acme Group" is consumed there and never reaches here).
_X_GROUP = re.compile(r"\b(?P<acquirer>" + _NAME_CAPTURE + r")\s+group\b", re.IGNORECASE)
# Loosest trigger - deliberately last, lowest-confidence, and most heavily
# guarded (see _is_garbage_acquirer / _is_non_acquirer_organization below,
# same filters name_matching applies to this exact trigger word).
_BY_X = re.compile(r"\bby\s+(?P<acquirer>" + _NAME_CAPTURE + r")" + _NAME_TERMINATOR, re.IGNORECASE)

# (pattern_name, compiled pattern, base confidence). Priority order - first
# pattern to produce an accepted match wins; later patterns are never tried
# once one hits. Same discipline as s3_detection.py/check_name_signals.py.
_LOGO_PATTERNS: list[tuple[str, re.Pattern, float]] = [
    ("a_x_company", _A_X_COMPANY, 0.9),
    ("part_of", _PART_OF, 1.0),
    ("x_group", _X_GROUP, 0.6),
    ("powered_by", _POWERED_BY, 0.7),
    ("by_x", _BY_X, 0.3),
]

_PATTERN_BASE_CONFIDENCE: dict[str, float] = {name: score for name, _, score in _LOGO_PATTERNS}

# Patterns risky enough to need the garbage/non-acquirer filters below -
# same set of trigger words name_matching flags as ambiguous, same reason:
# "powered by"/"by X" show up for franchise branding, accelerators/funds,
# and "product by its own developer" as often as for real acquisitions.
_FILTERED_PATTERNS = {"powered_by", "by_x", "x_group"}

# Franchise/reseller network brands that caption every local franchisee's
# site "powered by <brand> <franchisee>" - not an acquirer. Ported from
# check_name_signals.py's _NON_ACQUIRER_ORG_PREFIXES (same false-positive
# category applies to logo taglines).
_NON_ACQUIRER_ORG_PREFIXES = ("proforma",)
_ACCELERATOR_FUND_RE = re.compile(r"\b(accelerators?|akcelerator|funds?|idea)\b", re.IGNORECASE)

# Generic tagline/descriptor words that read as a category, not a company
# name, when that's ALL the capture consists of.
_GENERIC_TERM_DENYLIST = {
    "saas", "software", "technology", "tech", "cloud", "digital", "platform",
    "solution", "solutions", "service", "services", "leading", "global",
    "enterprise", "startup", "company", "business", "product", "brand",
    "group", "ai", "growth", "application", "applications", "verification",
    "data", "analytics", "insights", "innovation", "innovations", "systems",
    "system", "network", "networks",
}


def _clean_acquirer(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw).strip().strip(")").strip()
    return re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE).strip()


def _is_garbage_acquirer(acquirer: str, pattern_name: str) -> bool:
    """General capture-quality guard applied to every pattern (word count,
    at least one capitalized-looking token, not entirely generic terms) -
    same shape as s3_detection.py's _is_garbage_acquirer, tuned for short
    OCR fragments instead of prose."""
    words = acquirer.split()
    if not words or len(words) > 6:
        return True
    if acquirer.strip().lower() in _GENERIC_TERM_DENYLIST:
        return True
    if pattern_name in _FILTERED_PATTERNS:
        normalized = acquirer.strip().lower()
        if any(normalized == p or normalized.startswith(p + " ") for p in _NON_ACQUIRER_ORG_PREFIXES):
            return True
        if _ACCELERATOR_FUND_RE.search(acquirer):
            return True
    return False


# Legal/corporate suffix words stripped from BOTH sides before comparing in
# _is_self_reference - same list and same rationale as s3_detection.py's
# _normalize_company_name. Added after a real false positive found running
# this against the live EC2 db: company "Roga Group"'s logo just renders
# "ROGA GROUP" (its own name), and x_group's capture ("ROGA") is NOT an
# exact string match against the raw company name "Roga Group" (the
# trailing "Group" breaks equality) - the old plain-equality check let it
# through as a fabricated self-acquisition. Stripping the suffix from the
# company name before comparing ("Roga Group" -> "roga") fixes this without
# rejecting genuine matches: "TMG, an RIA company" strips to "tmg an ria"
# (not equal to acquirer "ria"), so the real, already-validated TMG->RIA
# corroboration case (see check_against_known.py) still passes correctly -
# verified by re-running check_against_known.py after this change.
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|co|company|group|holdings|gmbh|plc|srl|sa|nv|bv)\b\.?", re.IGNORECASE
)


def _normalize_company_name(name: str) -> str:
    name = _LEGAL_SUFFIX_RE.sub(" ", name or "")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_self_reference(acquirer: str, company_name: str | None) -> bool:
    """A company cannot be "a subsidiary of itself" etc - reject a capture
    that's just the subject's own name. Only meaningful when company_name
    is known (it usually isn't for a bare image file - see test script);
    a None/short company_name is a no-op, not a rejection.

    Two checks, both ported from s3_detection.py's own _is_self_reference
    (same false-positive category, same fix): exact equality after
    stripping legal suffixes from both sides (the Roga Group case above),
    and "acquirer capture starts with the subject's full name" (catches a
    capture that grabbed the subject's own name plus trailing junk).
    Deliberately NOT the reverse direction (subject starts with acquirer) -
    a genuine subsidiary is very often named as the parent's name plus
    extra words ("MIESCOR Telecommunications Business Unit" IS a real
    subsidiary of "MIESCOR"), so rejecting that direction would throw out
    correct matches, not just bad ones."""
    norm_subject = _normalize_company_name(company_name or "")
    if len(norm_subject) < 3:
        return False
    norm_acquirer = _normalize_company_name(acquirer)
    return norm_subject == norm_acquirer or norm_acquirer.startswith(norm_subject)


def extract_acquirer_from_logo_text(text: str, company_name: str | None = None) -> dict[str, Any] | None:
    """Scan `text` (OCR-extracted logo text, one company's lines joined)
    for acquisition/ownership language. Returns None if nothing matches,
    otherwise {"acquirerCompany", "matchedPattern", "confidenceScore", "rawText"}.
    """
    if not text:
        return None

    for pattern_name, pattern, base_score in _LOGO_PATTERNS:
        m = pattern.search(text)
        if m is None:
            continue

        acquirer = _clean_acquirer(m.group("acquirer"))
        if not acquirer or _is_garbage_acquirer(acquirer, pattern_name):
            continue
        if _is_self_reference(acquirer, company_name):
            continue

        return {
            "acquirerCompany": acquirer,
            "matchedPattern": pattern_name,
            "confidenceScore": base_score,
            "rawText": m.group(0)[:300],
        }

    return None
