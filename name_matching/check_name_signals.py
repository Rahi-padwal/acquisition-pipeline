"""Detect acquisitions from a company's NAME alone, via a narrow regex set."""

from __future__ import annotations
import re
from data_source import CompanyDataSource, CompanyRecord

# Capital letter + up to 50 more name-shaped chars.
_NAME_CAPTURE = r"[A-Z][\w&.,' -]{1,50}?"

# Stricter variant used only by powered_by/joined_forces_with/self_by_acquirer
# (the same three riskiest patterns already gated by _FILTERED_PATTERNS
# below). _NAME_CAPTURE's char class allows embedded periods/commas with no
# stopping point short of end-of-string, which is exactly what let a case
# like "TBWA Indonesia - Powered by AI. Driven by Disruption for Digital
# Growth." swallow the ENTIRE marketing tagline as the "acquirer" (nothing
# in the string was forbidden, so the capture had no reason to stop short of
# the end). This variant forbids crossing a ".", "," ";" or ":" that's
# followed by whitespace - a sentence/clause boundary - while still allowing
# the same punctuation when it's NOT followed by whitespace, so compact
# tokens like "Salary.com" or a trailing abbreviation like "Co." (period at
# the very end, nothing after it) still capture correctly. When a genuine
# acquirer name is followed by leftover prose, the pattern now simply fails
# to match at all (falls through to no match) instead of matching with a
# truncated or wrong acquirer.
_RISKY_ACQUIRER_CAPTURE = r"[A-Z](?:(?!\.\s|,\s|;\s|:\s)[\w&.,' -]){0,49}?"

# (name, pattern). Anchored end-to-end - the whole field is matched, not a
# sentence in a document. (?i:...) scopes case-insensitivity to keywords only,
# so the acquirer's [A-Z] anchor still requires a real capital letter.
_NAME_ACQUIRER_PATTERNS = [
    (
        "subsidiary_of",
        re.compile(
            # [\s-]+ so "wholly-owned" (hyphenated) matches too.
            r"^(?P<self>.+?),?\s+(?i:(?:is\s+)?an?\s+(?:wholly[\s-]+owned[\s-]+)?subsidiary\s+of)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "owned_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:is\s+)?owned\s+by)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "part_of",
        re.compile(
            r"^(?P<self>.+?)[,\-–—]?\s*(?i:part\s+of)\s+(?i:the\s+)?"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")(?:\s+(?i:group))?\)?\s*$"
        ),
    ),
    (
        "a_x_company",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+(?i:company)\)?\s*$"
        ),
    ),
    (
        # Reverse word order from "a_x_company": "X, a company of Y".
        "a_company_of",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?\s+company\s+of)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "a_x_brand",
        re.compile(
            r"^(?P<self>.+?)[,(]\s*(?i:an?)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\s+(?i:brand)\)?\s*$"
        ),
    ),
    (
        "joined_forces_with",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:has\s+)?joined\s+forces\s+with)\s+"
            r"(?P<acquirer>" + _RISKY_ACQUIRER_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "acquired_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:(?:(?:has\s+been|was|is)\s+)?acquired\s+by)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "powered_by",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:powered\s+by)\s+(?P<acquirer>" + _RISKY_ACQUIRER_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "operating_unit_of",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:an?\s+operating\s+unit\s+of)\s+"
            r"(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        "division_of",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:an?\s+division\s+of)\s+(?P<acquirer>" + _NAME_CAPTURE + r")\)?\s*$"
        ),
    ),
    (
        # Reversed group roles: acquirer comes BEFORE "trading as", self after.
        "trading_as",
        re.compile(
            r"^(?P<acquirer>" + _NAME_CAPTURE + r"),?\s+(?i:trading\s+as)\s+(?P<self>.+?)\)?\s*$"
        ),
    ),
    (
        # Reversed roles again: "Google acquires Nest" - acquirer stated first.
        "acquirer_acquires_self",
        re.compile(
            r"^(?P<acquirer>" + _NAME_CAPTURE + r"),?\s+(?i:acquires)\s+(?P<self>.+?)\)?\s*$"
        ),
    ),
    (
        # "Nest by Google" - riskier, "by" is common outside acquisitions too.
        "self_by_acquirer",
        re.compile(
            r"^(?P<self>.+?),?\s+(?i:by)\s+(?P<acquirer>" + _RISKY_ACQUIRER_CAPTURE + r")\)?\s*$"
        ),
    ),
]

# Unanchored - scans a parenthetical fragment inside a larger string, e.g.
# "Series A - Advizzo (part of Calisen Group)".
_PARENTHETICAL_PART_OF = re.compile(r"\((?i:part\s+of\s+(?:the\s+)?)(?P<acquirer>" + _NAME_CAPTURE + r")\)")

# Generic filler a match can capture instead of a real company name.
_GENERIC_TERM_DENYLIST = {
    "saas",
    "software",
    "technology",
    "tech",
    "cloud",
    "digital",
    "platform",
    "solution",
    "solutions",
    "service",
    "services",
    "leading",
    "global",
    "enterprise",
    "startup",
    "company",
    "business",
    "product",
    "brand",
    "group",
}


def _names_match(a: str, b: str) -> bool:
    """Case-insensitive equality, treating one name containing the other as a match."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    return a in b or b in a


# Legal-entity suffixes and generic descriptors stripped by _company_identity_match()
# before comparing two company names - "fabrikado GmbH" vs "fabrikado.com" or
# "New Era Financial Group Inc" vs "New Era Financial Advisors" are the same
# company under everyday naming variance, not a mismatch.
_LEGAL_SUFFIX_WORDS = {
    "inc", "llc", "llp", "ltd", "limited", "corp", "corporation", "co", "company",
    "gmbh", "group", "grp", "plc", "srl", "sa", "ag", "pvt", "private", "ab",
    "nv", "bv", "sarl", "kg", "oy", "as",
}
_DOMAIN_SUFFIX_RE = re.compile(r"\.(com|io|co|net|org)\b")
_NON_WORD_RE = re.compile(r"[^\w\s]")


def _normalize_company_identity(name: str) -> str:
    text = _DOMAIN_SUFFIX_RE.sub("", (name or "").lower())
    text = _NON_WORD_RE.sub(" ", text)
    tokens = text.split()
    while tokens and tokens[-1] in _LEGAL_SUFFIX_WORDS:
        tokens.pop()
    return " ".join(tokens)


def _company_identity_match(a: str, b: str) -> bool:
    """Looser than _names_match(): strips legal-entity suffixes/domain TLDs
    and punctuation before comparing, and also tries a whitespace-free
    comparison (catches "Wizco Media" vs "Wizcomedia"-style spacing drift).
    Used to sanity-check that a structured acquiredBy record's
    acquireeCompanyName actually refers to the company it's attached to -
    see _check_structured_fields()."""
    norm_a, norm_b = _normalize_company_identity(a), _normalize_company_identity(b)
    if not norm_a or not norm_b:
        return False
    if norm_a in norm_b or norm_b in norm_a:
        return True
    compact_a, compact_b = norm_a.replace(" ", ""), norm_b.replace(" ", "")
    return len(compact_a) >= 3 and len(compact_b) >= 3 and (compact_a in compact_b or compact_b in compact_a)


def _is_valid_acquirer_candidate(candidate: str, self_name: str) -> bool:
    candidate = candidate.strip()
    if not (3 <= len(candidate) <= 50):
        return False
    if not candidate[0].isupper():
        return False
    if candidate.lower() in _GENERIC_TERM_DENYLIST:
        return False
    if _names_match(candidate, self_name):
        return False
    return True


# --- powered_by / joined_forces_with-specific quality filters ----------------
#
# These two patterns are the weakest, riskiest triggers in _NAME_ACQUIRER_PATTERNS
# (see their 0.7 base confidence below) - "X powered by Y" and "X joined forces
# with Y" both use ambiguous everyday phrasing that shows up for reasons other
# than "Y acquired X": franchise/reseller-network branding, accelerator/grant
# relationships, and prose the capture over-grabbed instead of isolating a
# real company name. Manual review of ~40 real matches turned up two
# recurring, nameable failure categories - see the two checks below. Neither
# adjusts confidence score; a match that trips either filter is discarded
# entirely (not returned), same as a self-truncation rejection.
#
# Deliberately scoped to ONLY these two patterns via the pattern_name check at
# each call site - was_acquired_by, subsidiary_of, acquired_by, etc. are much
# more explicit/unambiguous phrasing and should not be filtered by this logic.

# "Proforma" is a promotional-products franchise brand: independent local
# franchisees caption their sites "powered by Proforma <Franchisee Name>",
# which reads exactly like an acquisition but isn't one - matched as a
# prefix, not just the bare word, since the captured acquirer is routinely
# "Proforma Albrecht & Co." / "Proforma <whoever>", not literally "Proforma"
# alone. AXI was investigated and rejected for this list - see
# combined_matching conversation history: AXI's own press release names
# InterIT, IT Provider and DVIT as genuine acquisitions (all three brands of
# ITP Group), not franchisees, so it stays out.
_NON_ACQUIRER_ORG_PREFIXES = ("proforma",)

# Accelerators, grant programs and investment funds back/support companies;
# they don't acquire them. Matched as a standalone word (not substring) so
# this doesn't reject a real company name that merely contains one of these
# as a word-fragment (e.g. "Idealab" is untouched; "NC IDEA" is caught).
_ACCELERATOR_FUND_RE = re.compile(r"\b(accelerators?|akcelerator|funds?|idea)\b", re.IGNORECASE)


def _is_non_acquirer_organization(candidate: str) -> bool:
    """True when `candidate` names a known category of non-acquirer
    organization (franchise network, accelerator, fund) rather than a real
    acquirer - see the constants above."""
    normalized = candidate.strip().lower()
    if any(normalized == prefix or normalized.startswith(prefix + " ") for prefix in _NON_ACQUIRER_ORG_PREFIXES):
        return True
    return bool(_ACCELERATOR_FUND_RE.search(candidate))


# Sentence-filler words that show up when the regex over-captured surrounding
# prose instead of isolating a company name - "joined forces with leading
# health care providers and major U.S. ..." being the motivating case.
_ACQUIRER_FILLER_PHRASE_RE = re.compile(
    r"\b(?:leading|major|other|various|several|top|well[\s-]known|renowned|trusted)\b|\b(?:like|such\s+as|including)\b",
    re.IGNORECASE,
)

# Generic/tagline words that read as descriptive terms rather than a company
# name when that's ALL the capitalized content in a capture amounts to (e.g.
# "AI Growth", "Application Verification"). Deliberately a separate set from
# _GENERIC_TERM_DENYLIST above (which is shared by every pattern) - this one
# only feeds _is_garbage_acquirer_text() and only for powered_by/
# joined_forces_with, so widening it can't affect any other pattern's scoring.
_POWERED_BY_JOINED_FORCES_GENERIC_WORDS = {
    "ai", "growth", "application", "applications", "verification",
    "solution", "solutions", "platform", "platforms", "software",
    "technology", "technologies", "digital", "data", "analytics",
    "insights", "innovation", "innovations", "systems", "system",
    "service", "services", "cloud", "network", "networks",
}


def _is_garbage_acquirer_text(candidate: str) -> bool:
    """True when `candidate` looks like over-captured prose rather than a
    company name: either it contains a sentence-filler word/phrase, or none
    of its capitalized-looking words are anything but generic tagline terms
    (so there's no actual proper-noun-shaped content at all)."""
    if _ACQUIRER_FILLER_PHRASE_RE.search(candidate):
        return True

    capitalized_words = [w.strip(".,'-&") for w in candidate.split() if w[:1].isupper()]
    capitalized_words = [w for w in capitalized_words if w]
    if not capitalized_words:
        return True
    return all(w.lower() in _POWERED_BY_JOINED_FORCES_GENERIC_WORDS for w in capitalized_words)


# self_by_acquirer ("X by Y") gets both guards above (it's exactly as
# ambiguous a trigger as powered_by/joined_forces_with - the code's own
# comment on the pattern already calls it out as riskier) PLUS one more,
# below: "X by Y" also fires on "Product by its own Developer" branding
# (e.g. "NimbleS2P by Techpanion", "MPO Solution by Ngenio") where Y built
# and owns X from the start - never acquired anything - which the acquirer-
# side checks above can't catch since the acquirer name itself (Techpanion,
# Ngenio) is a perfectly real, non-generic company name. The signal here is
# on the SELF side instead: a self_name containing a product/tool-shaped
# word is a tell that "self" names a product, not the acquired company.
#
# Known limitation, accepted rather than chased: this only catches
# space-separated product words ("MPO Solution" -> catches "Solution").
# A compound, no-space brand like "NimbleS2P" has no word boundary around
# the embedded "S2P" for this check to find, so that shape still gets
# through - a camelCase/acronym splitter to close that gap was considered
# and skipped as not worth the fragility it would add.
_PRODUCT_LIKE_SELF_WORDS = {
    "solution", "solutions", "platform", "platforms", "suite", "system",
    "systems", "software", "app", "apps", "tool", "tools", "module",
    "engine", "portal", "dashboard",
}


def _is_product_like_self_name(self_name: str) -> bool:
    words = [w.strip(".,()'\"-").lower() for w in self_name.split()]
    return any(w in _PRODUCT_LIKE_SELF_WORDS for w in words if w)


_FILTERED_PATTERNS = {"powered_by", "joined_forces_with", "self_by_acquirer"}


def _is_self_truncation(candidate: str, self_name: str) -> bool:
    """True when `candidate` is just `self_name` with a short trailing
    suffix/acronym chopped off (e.g. candidate "Moore" vs self name
    "Moore RMG") - a real acquirer is a distinct company, never a
    truncation of the subject's own name.

    Deliberately narrower than "candidate appears anywhere in self_name":
    that would also flag legitimate cases where a company's own (long,
    descriptive) name happens to end with its real acquirer's name, e.g.
    "Authomize, now part of Delinea" genuinely contains "Delinea" - see
    _check_structured_fields()'s docstring. Only a PREFIX match with a
    short (<=2 word) remainder counts as a truncation, which targets the
    "self name = candidate + suffix/acronym" shape specifically, without
    rejecting a candidate that merely shows up later in a longer sentence.
    """
    candidate_norm = candidate.strip().lower()
    self_norm = (self_name or "").strip().lower()
    if not candidate_norm or not self_norm:
        return False
    if candidate_norm == self_norm:
        return True
    if not self_norm.startswith(candidate_norm):
        return False
    remainder = self_norm[len(candidate_norm) :].strip(" ,.-&")
    return bool(remainder) and len(remainder.split()) <= 2 and len(remainder) <= 15


# --- Confidence scoring -------------------------------------------------------
#
# Base score per matched_pattern, 0.0-1.0. This is the only place these
# numbers live - adjust here, not per call site.
_PATTERN_BASE_CONFIDENCE: dict[str, float] = {
    "subsidiary_of": 1.0,               # "a wholly-owned subsidiary of X"
    "owned_by": 1.0,                    # "owned by X" - same explicit-ownership tier
    "part_of": 1.0,                     # "(now) part of X"
    "acquired_by": 1.0,                 # "has been acquired by X"
    "joined_forces_with": 1.0,          # "joined forces with X"
    "operating_unit_of": 1.0,           # "an operating unit of X"
    "division_of": 1.0,                 # "a division of X"
    "acquirer_acquires_self": 1.0,      # "X acquires Company"
    "a_x_company": 0.9,                 # "a X company"
    "a_company_of": 0.9,                # "a company of X" - reverse of a_x_company, same family
    "a_x_brand": 0.9,                   # "a X brand" - same family as a_x_company
    "trading_as": 0.9,                  # "X, trading as Company"
    # "Company by X" - demoted from 0.9 after manual review found 7/9 sampled
    # matches were false positives (X is the product's own creator/developer,
    # not an acquirer - e.g. "CoLumbo by Smart Soft Healthcare", "BeTalent by
    # Zircon"). That's a real-world fact ("did Y buy X, or did Y build X?"),
    # not a text-shape signal a regex can resolve on its own, so - unlike
    # powered_by/joined_forces_with above - this isn't a case a garbage-text
    # filter can fix; the low score instead lets combined_check.py's cross-
    # source corroboration logic do the real work (bump back up when
    # structured_field/description independently agree, stay low otherwise).
    "self_by_acquirer": 0.3,
    "powered_by": 0.7,                  # "powered by X" - weakest explicit regex pattern
    "structured_field": 1.0,            # ES's own isAcquired/acquiredBy fields; see modifier below
    "funding_round_parenthetical": 0.7,  # "(part of X)" scraped from a funding-round label
}


def _base_confidence_for_pattern(matched_pattern: str) -> float:
    return _PATTERN_BASE_CONFIDENCE.get(matched_pattern, 0.5)


def _acquirer_corroborated_in_own_text(acquirer: str, raw: dict) -> bool:
    """Modifier: does `acquirer` appear anywhere in the company's own
    companyName, description, or linkedinProfileName? Used to sanity-check
    structured-field acquirer names specifically - fundingAndInvestment.
    acquiredBy sometimes holds a mislinked/chained transaction (see
    _check_structured_fields()'s docstring); if the acquirer's name is
    nowhere in the company's own text, that's a corroboration gap worth
    flagging even after the acquiree-name guard already let it through."""
    acquirer_norm = acquirer.strip().lower()
    if not acquirer_norm:
        return False
    haystacks = (raw.get("companyName"), raw.get("description"), raw.get("linkedinProfileName"))
    return any(acquirer_norm in (haystack or "").lower() for haystack in haystacks)


def _score_match(match: dict, raw: dict) -> dict:
    """Attach confidence_score/confidence_notes to `match` (mutates and
    returns it). Only the structured-field-uncorroborated modifier is
    applied here, since it only needs this one record's own raw text.
    Cross-source modifiers (corroboration/conflict against the description-
    based path) need that path's output too and live one level up, in
    combined_matching/combined_check.py, the one place both detectors'
    results are visible together."""
    matched_pattern = match["matched_pattern"]
    score = _base_confidence_for_pattern(matched_pattern)
    notes = None
    if matched_pattern == "structured_field" and not _acquirer_corroborated_in_own_text(
        match["candidate_name"], raw
    ):
        score = min(score, 0.4)
        notes = "structured_field_uncorroborated"
    match["confidence_score"] = score
    match["confidence_notes"] = notes
    return match


def find_acquirer_in_name(company_name: str) -> dict | None:
    """Return {candidate_name, matched_pattern, self_name, raw_name} for the
    first valid match, or None."""

    name = (company_name or "").strip()
    if not name:
        return None

    for pattern_name, pattern in _NAME_ACQUIRER_PATTERNS:
        match = pattern.match(name)
        if not match:
            continue

        acquirer = match.group("acquirer").strip().rstrip(").,")
        self_name = match.group("self").strip()
        if not _is_valid_acquirer_candidate(acquirer, self_name):
            continue
        if pattern_name in _FILTERED_PATTERNS and (
            _is_non_acquirer_organization(acquirer) or _is_garbage_acquirer_text(acquirer)
        ):
            # `return None`, not `continue`: self_by_acquirer ("X by Y") is a
            # later, looser fallback in this same pattern list, and "powered
            # by X" / "joined forces with X" text always ALSO matches "X by
            # Y" - falling through would just let self_by_acquirer re-catch
            # the exact text this filter just rejected. Other detectors
            # (structured_field, funding-round parenthetical, description-
            # based) are untouched and still run separately in
            # check_company_record().
            return None
        if pattern_name == "self_by_acquirer" and _is_product_like_self_name(self_name):
            return None

        return {
            "candidate_name": acquirer,
            "matched_pattern": pattern_name,
            "self_name": self_name,
            "raw_name": name,
        }

    return None


def _check_structured_fields(raw: dict, record_company_id: str) -> dict | None:
    """Prefer Elastic's own isAcquired / acquirerCompanyName fields over
    regex-parsing the name, when present.

    Self-reference is guarded via acquirerCompanyId vs the record's own
    companyId, not a name substring check - companyName routinely already
    contains the acquirer's name (e.g. "Authomize, now part of Delinea").

    Also guarded against a chained-acquisition attachment error seen
    upstream: fundingAndInvestment.acquiredBy sometimes holds a LATER,
    unrelated transaction that has nothing to do with this record (e.g. a
    company that was itself acquired, whose subsequent-acquisition-by-a-
    third-party ends up linked here instead of its own acquisition). The
    tell is acquireeCompanyName not resembling this record's own
    companyName at all - a genuine acquiredBy entry for THIS company should
    name THIS company (or a recognizable past name of it) as the acquiree.
    When it doesn't, don't trust acquirerCompanyName either; returning None
    here lets the caller fall through to name-regex / description-based
    detection instead of returning a confident but wrong acquirer.
    """
    if not raw.get("isAcquired"):
        return None

    acquired_by = (raw.get("fundingAndInvestment") or {}).get("acquiredBy") or {}
    # Nested fields can come back as a list even for a single value.
    if isinstance(acquired_by, list):
        acquired_by = acquired_by[0] if acquired_by else {}

    acquirer = (acquired_by.get("acquirerCompanyName") or "").strip()
    acquirer_id = str(acquired_by.get("acquirerCompanyId") or "")
    if not acquirer:
        return None
    if acquirer_id and acquirer_id == str(record_company_id):
        return None

    company_name = raw.get("companyName", "")
    acquiree_name = (acquired_by.get("acquireeCompanyName") or "").strip()
    if acquiree_name and not _company_identity_match(acquiree_name, company_name):
        return None

    return {
        "candidate_name": acquirer,
        "matched_pattern": "structured_field",
        "self_name": company_name,
        "raw_name": company_name,
    }


def _check_funding_milestones(raw: dict, self_name: str) -> dict | None:
    """Fallback for stale/empty isAcquired+acquiredBy: scans funding-round
    names for a "(part of X)" parenthetical, e.g. "Series A - Advizzo (part
    of Calisen Group)"."""
    funding = raw.get("fundingAndInvestment") or {}
    if isinstance(funding, list):
        funding = funding[0] if funding else {}

    milestones = funding.get("fundingMilestonesTimeseries") or []
    if isinstance(milestones, dict):
        milestones = [milestones]

    for milestone in milestones:
        if not isinstance(milestone, dict):
            continue
        funding_round = milestone.get("fundingRound")
        texts = funding_round if isinstance(funding_round, list) else [funding_round]

        for text in texts:
            match = _PARENTHETICAL_PART_OF.search(text or "")
            if not match:
                continue
            acquirer = match.group("acquirer").strip().rstrip(").,")
            if not _is_valid_acquirer_candidate(acquirer, self_name):
                continue
            return {
                "candidate_name": acquirer,
                "matched_pattern": "funding_round_parenthetical",
                "self_name": self_name,
                "raw_name": text,
            }

    return None


def check_company_record(record) -> dict | None:
    """Check a CompanyRecord for an acquirer: structured fields, then name
    regex, then funding-round parenthetical, in that order.

    Every candidate from every detector is run through
    _is_self_truncation() against that detector's own notion of "self"
    before being accepted, regardless of which one produced it - this is
    the one place all three paths funnel through, so it's the one place a
    truncation-of-own-name candidate (e.g. "Moore" from "Moore RMG") gets
    caught no matter which detector let it slip past its own checks (or,
    in _check_structured_fields()'s case, had no such check at all).
    """
    for detector in (
        lambda: _check_structured_fields(record.raw, record.company_id),
        lambda: find_acquirer_in_name(record.company_name),
        lambda: _check_funding_milestones(record.raw, record.company_name),
    ):
        match = detector()
        if match is None:
            continue
        # Rejection takes priority over scoring: a self-truncated candidate
        # never reaches _score_match() - it's `continue`d past entirely, not
        # scored low and returned.
        if _is_self_truncation(match["candidate_name"], match["self_name"]):
            continue
        return _score_match(match, record.raw)

    return None


def _domain_key(record: CompanyRecord) -> str:
    return record.raw.get("companyWebsiteDomain", "")


def _search_and_check(
    query: str,
    records: list[CompanyRecord],
    key=lambda r: r.company_name,
) -> dict:
    """A name/domain search can return several different companies (e.g. many
    sub-brands sharing one parent domain). Check EVERY candidate and return
    the first with real evidence, trying an exact match on `key` first, then
    result order. Falls back to "not acquired" only if none have evidence.
    """
    query_lower = query.strip().lower()
    exact = [r for r in records if key(r).strip().lower() == query_lower]
    ordered = exact + [r for r in records if r not in exact]

    for record in ordered:
        match = check_company_record(record)
        if match is not None:
            return _positive_result(record, match)

    return _not_acquired_result(ordered[0])


def _positive_result(record: CompanyRecord, match: dict) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": match["candidate_name"],
        "confidence_score": match.get("confidence_score"),
        "confidence_notes": match.get("confidence_notes"),
        "evidence": [
            {
                "candidate_name": match["candidate_name"],
                "matched_pattern": match["matched_pattern"],
                "raw_name": match["raw_name"],
            }
        ],
        "status": "checked",
    }


def _not_found_result(query: str) -> dict:
    return {
        "company_id": None,
        "company_name": query,
        "is_acquired": False,
        "likely_acquirer": None,
        "confidence_score": None,
        "confidence_notes": None,
        "evidence": [],
        "status": "not_found",
    }


def _not_acquired_result(record: CompanyRecord) -> dict:
    return {
        "company_id": record.company_id,
        "company_name": record.company_name,
        "is_acquired": False,
        "likely_acquirer": None,
        "confidence_score": None,
        "confidence_notes": None,
        "evidence": [],
        "status": "checked",
    }


def run_name_search_check(company_name: str, data_source: CompanyDataSource) -> dict:
    """Search `data_source` for `company_name` and check the best match for
    an acquirer. See _search_and_check() for how multi-hit results are
    resolved; status is "checked" or "not_found"."""
    results = data_source.search_by_name(company_name)
    if not results:
        return _not_found_result(company_name)

    return _search_and_check(company_name, results)


def run_domain_search_check(company_domain: str, data_source: CompanyDataSource) -> dict:
    """Same as run_name_search_check(), keyed by companyWebsiteDomain. ES's
    `match` query is analyzed, not exact, so results are filtered down to
    ones whose domain genuinely contains the query first."""
    results = data_source.search_by_domain(company_domain)
    query_lower = company_domain.strip().lower()
    results = [r for r in results if query_lower in _domain_key(r).lower()]
    if not results:
        return _not_found_result(company_domain)

    return _search_and_check(company_domain, results, key=_domain_key)


def run_dwh_id_check(dwh_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `dwh_id` (a unique key) and check it for an acquirer. No
    disambiguation needed since dwhId identifies exactly one record."""
    record = data_source.get_by_dwh_id(dwh_id)
    if record is None:
        return _not_found_result(dwh_id)

    match = check_company_record(record)
    if match is None:
        return _not_acquired_result(record)

    return _positive_result(record, match)


def run_name_check(company_id: str, data_source: CompanyDataSource) -> dict:
    """Look up `company_id` and check its companyName for an embedded acquirer."""
    record = data_source.get_by_id(company_id)
    if record is None:
        return {
            "company_id": company_id,
            "company_name": None,
            "is_acquired": False,
            "likely_acquirer": None,
            "confidence_score": None,
            "confidence_notes": None,
            "evidence": [],
            "status": "not_found",
        }

    match = find_acquirer_in_name(record.company_name)
    if match is None:
        return {
            "company_id": company_id,
            "company_name": record.company_name,
            "is_acquired": False,
            "likely_acquirer": None,
            "confidence_score": None,
            "confidence_notes": None,
            "evidence": [],
            "status": "checked",
        }

    # find_acquirer_in_name() only ever produces companyName-regex matches
    # (never structured_field), so the structured-field-uncorroborated
    # modifier never applies here - a plain base-score lookup is enough.
    confidence_score = _base_confidence_for_pattern(match["matched_pattern"])

    return {
        "company_id": company_id,
        "company_name": record.company_name,
        "is_acquired": True,
        "likely_acquirer": match["candidate_name"],
        "confidence_score": confidence_score,
        "confidence_notes": None,
        "evidence": [
            {
                "candidate_name": match["candidate_name"],
                "matched_pattern": match["matched_pattern"],
                "raw_name": match["raw_name"],
            }
        ],
        "status": "checked",
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python check_name_signals.py '<company name string>'")
        sys.exit(1)

    result = find_acquirer_in_name(sys.argv[1])
    print(result if result else "No acquirer detected in that name.")
