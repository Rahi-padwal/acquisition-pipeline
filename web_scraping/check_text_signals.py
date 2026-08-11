"""Detect acquisition signals (e.g. "was acquired by X", "our parent company
X") in a company's rendered website text.

Pipeline: get_tiered_target_urls -> fetch_page -> extract_text ->
check_acquirer_match, wired together by run_check().

Candidate sub-URLs are checked in tiers rather than one flat, homepage-first
list:
  Tier 1 (high signal):    /press /press-releases /newsroom /about /our-story
  Tier 2 (legal, lower noise): /terms /legal /privacy /privacy-policy
                                /terms-of-service /tos
  Tier 3 (high noise):     the homepage
The homepage is checked LAST and only if tiers 1-2 produced no evidence,
since it's the page most likely to carry unrelated third-party badges/
widgets that can look like an acquisition mention but aren't.

This fixed list is the entire search space - with one bounded exception:
press/news listing pages (e.g. /press-releases) are themselves just an
index, so individual articles are one level below. From a listing page
only, a small, keyword-filtered set of same-domain links (slug containing
"acqui", "merger", etc.) are followed - see _extract_acquisition_leaning_links().
This is not general crawling: it never follows links found on a non-listing
page, and it's capped at _MAX_LISTING_SUBPAGES per listing page.

Matching is deliberately STRICT: it is better to report "not acquired" than
to falsely flag a company. check_acquirer_match() only fires on a small set
of factual-acquisition phrasings, skips anything wrapped in hypothetical/
conditional language ("may be", "in the event of", ...), and runs every
candidate through a denylist + self-reference + shape check before
accepting it. See check_acquirer_match()'s docstring for the full rationale
- this replaced an earlier, looser NLP-assisted approach after it flagged
"Registration Data" (WHOIS/registrar boilerplate) as an acquirer.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

FETCH_TIMEOUT_MS = 10_000
FETCH_RETRIES = 1  # number of retries after the first attempt, for real failures (not 404s)

# A realistic desktop Chrome UA + headers, so CDN bot protection (Cloudflare,
# Akamai, etc.) is less likely to serve a challenge/empty shell instead of
# the real page.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
_EXTRA_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Mode": "navigate",
}
_VIEWPORT = {"width": 1280, "height": 800}

# Candidate paths, grouped by signal tier. Tier 3 (homepage) is only
# consulted if tiers 1 and 2 produce no evidence — see module docstring.
_TIER_1_PATHS = ["/press", "/press-releases", "/newsroom", "/about", "/our-story"]
_TIER_2_PATHS_RAW = ["/terms", "/legal", "/privacy", "/privacy-policy", "/terms-of-service", "/tos"]
_TIER_2_PATHS = list(dict.fromkeys(_TIER_2_PATHS_RAW))  # dedupe, preserve order
_TIER_3_PATHS = [""]  # homepage

# Press/news listing pages (e.g. /press-releases) are themselves just an
# index of individual articles - the acquisition announcement usually lives
# one level below, as its own article. From a listing page only, we follow
# same-domain links whose slug looks acquisition-related; this is bounded
# and keyword-filtered, not general crawling.
_LISTING_PATHS = {"/press", "/press-releases", "/press-coverage", "/newsroom", "/news", "/blog"}
_ACQUISITION_SLUG_KEYWORDS = (
    "acqui",  # covers acquisition/acquires/acquired/acquiring
    "merger",
    "merges",
    "buyout",
    "private-equity",
    "go-private",
    "goes-private",
    "take-private",
    "tender-offer",
    "joins-forces",
)
_MAX_LISTING_SUBPAGES = 15

# --- Strict acquisition matching -------------------------------------------
#
# Name capture: capital letter, then up to 50 more "name-shaped" chars
# (letters/digits/&/./,/'/space/hyphen). This is intentionally narrower than
# free-form prose text, so it can't run on into an unrelated clause the way
# a bare ".+?" would.
_NAME_CAPTURE = r"[A-Z][\w&.,' -]{1,50}"

# name, pattern-id pairs (case-SENSITIVE - the [A-Z] anchor only means what
# it says if we don't let IGNORECASE fold it back into matching lowercase).
# \s+ (not literal single spaces) between keywords: real HTML text nodes
# can carry embedded newlines/indentation from pretty-printed markup, which
# a literal " " would fail to bridge.
_STRICT_ACQUIRER_PATTERNS = [
    ("subsidiary_of", re.compile(r"is\s+a\s+(?:wholly\s+owned\s+)?subsidiary\s+of\s+(" + _NAME_CAPTURE + r")")),
    ("division_of", re.compile(r"(?:is\s+|,\s*)?a\s+division\s+of\s+(" + _NAME_CAPTURE + r")")),
    ("part_of", re.compile(r"is\s+part\s+of\s+(?:the\s+)?(" + _NAME_CAPTURE + r")(?:\s+(?:group|family))?")),
    ("was_acquired_by", re.compile(r"was\s+acquired\s+by\s+(" + _NAME_CAPTURE + r")")),
    ("acquired_by_year", re.compile(r"acquired\s+by\s+(" + _NAME_CAPTURE + r")\s+in\s+\d{4}")),
    ("parent_company", re.compile(r"our\s+parent\s+company,?\s+(" + _NAME_CAPTURE + r")")),
    (
        "ownership_of_parent",
        re.compile(
            r"under\s+(?:the\s+)?(?:common\s+)?ownership\s+(?:or\s+control\s+)?of\s+(?:our\s+)?(?:ultimate\s+)?"
            r"parent\s+company,?\s+(" + _NAME_CAPTURE + r")"
        ),
    ),
    ("a_x_company", re.compile(r"an?\s+(" + _NAME_CAPTURE + r")\s+company")),
    # Softer ownership language - real, but weaker evidence than the patterns
    # above (an investor "backing" a company, or a "brand"/"product" label,
    # isn't as unambiguous a claim as "is a subsidiary of" or "was acquired
    # by"). See _PATTERN_CONFIDENCE.
    ("backed_by", re.compile(r"backed\s+by\s+(" + _NAME_CAPTURE + r")")),
    ("a_x_brand", re.compile(r"an?\s+(" + _NAME_CAPTURE + r")\s+brand")),
    ("owned_by", re.compile(r"is\s+owned\s+by\s+(" + _NAME_CAPTURE + r")")),
    ("a_x_product", re.compile(r"a\s+(" + _NAME_CAPTURE + r")\s+product")),
]

# Per-pattern confidence tier for the final result (see run_check()). The
# explicit "acquired by" / "subsidiary of" / "division of" / "parent
# company" families are unambiguous ownership claims -> high. "part_of" and
# "ownership_of_parent" are grammatically parallel to those (same kind of
# direct corporate-structure statement), so grouped with them. The newer,
# softer phrasings (backed by / a X brand / owned by / a X product, plus
# the pre-existing "a X company") read as real but less certain evidence,
# so they're capped at medium even when they're the only match.
_PATTERN_CONFIDENCE = {
    "subsidiary_of": "high",
    "division_of": "high",
    "part_of": "high",
    "was_acquired_by": "high",
    "acquired_by_year": "high",
    "parent_company": "high",
    "ownership_of_parent": "high",
    "a_x_company": "medium",
    "backed_by": "medium",
    "a_x_brand": "medium",
    "owned_by": "medium",
    "a_x_product": "medium",
}
_CONFIDENCE_RANK = {"high": 2, "medium": 1}

# Skip a match if any of these appear in the ~15 words around it - standard
# "if we are ever acquired" privacy-policy boilerplate is hypothetical, not
# evidence of an actual acquisition.
_HYPOTHETICAL_MARKERS = (
    "in the event of",
    "in the case of",
    "may be",
    "should we",
    "if we are",
    "in connection with any future",
    "potential",
    "were to be",
)

# "an? X company" additionally skips if the text nearby is describing the
# site's own independence ("we are ... company", "is an independent ...
# company"), not naming an acquirer.
_SELF_DESCRIPTION_MARKERS = ("we are", "is an independent")

# Case-insensitive; a candidate containing (or contained in) any of these is
# rejected outright. Mostly WHOIS/registrar boilerplate and generic legal-
# entity-form filler that regexes above can mistake for a company name -
# this is what the "Registration Data" bug (Whimsical) fell into.
_DENYLIST = {
    "registration data",
    "registrant",
    "registrar",
    "whois",
    "domains by proxy",
    "godaddy",
    "data controller",
    "data protection officer",
    "supervisory authority",
    "regulatory authority",
    "the company",
    "this company",
    "our company",
    "the site",
    "the app",
    "equal opportunity",
    "service provider",
    "third party",
    "third-party",
}

# Common business/industry descriptors that a greedy match can capture on
# their own (e.g. "backed by a leading SaaS provider" -> candidate "SaaS")
# and mistake for an actual company name. Checked as an EXACT match against
# the full (lowercased) candidate, not substring containment like _DENYLIST
# above - a real brand name that happens to contain one of these words as
# part of a longer name (or a short acronym like "VWO") must not be
# rejected just for that.
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
    "data",
}

# Trailing words/punctuation that terminate a captured company name.
_STOP_WORDS = r"and|since|in|to|for|as|which|that|with|a|an|the"
_TRAILING_CUTOFF = re.compile(
    r"[.,;:!?\n\r]|(?:\s+(?:" + _STOP_WORDS + r")\b)", re.IGNORECASE
)

# "© 2024 Acme Inc.", "Copyright 2024 Acme Inc" - used only for extract_text's
# "copyright" field; check_acquirer_match no longer matches off of it (kept
# narrow and separate from the strict acquirer patterns above).
_CURRENT_YEAR_PLACEHOLDER = r"(?:19|20)\d{2}"
_CORP_SUFFIX = r"(?:Inc|LLC|L\.L\.C\.?|Ltd|Corp|Corporation|Limited)\.?"
_COPYRIGHT_PATTERN = re.compile(
    r"(?:©|\(c\)|copyright)\s*(?:©\s*)?(?:" + _CURRENT_YEAR_PLACEHOLDER + r")"
    r"(?:\s*[-–]\s*(?:" + _CURRENT_YEAR_PLACEHOLDER + r"))?\s+"
    r"([^.;:!?\n\r]{1,80}?,?\s+" + _CORP_SUFFIX + r")(?:[.,;:!?\n\r]|$)",
    re.IGNORECASE,
)


def derive_company_name(url: str) -> str:
    """Derive a display name from a URL's domain, e.g. 'www.acme.com' -> 'Acme'."""
    netloc = urlparse(url).netloc or urlparse(url).path
    netloc = netloc.split(":")[0]  # drop port, if any
    if netloc.startswith("www."):
        netloc = netloc[4:]
    label = netloc.split(".")[0] if netloc else url
    label = re.sub(r"[-_]+", " ", label).strip()
    return label.title() if label else url


def get_tiered_target_urls(base_url: str) -> dict[str, list[str]]:
    """Build the candidate URLs for `base_url`, grouped by signal tier."""
    parsed = urlparse(base_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path.split("/")[0]
    root = f"{scheme}://{netloc}"
    return {
        "tier1": [root + path for path in _TIER_1_PATHS],
        "tier2": [root + path for path in _TIER_2_PATHS],
        "tier3": [root + path for path in _TIER_3_PATHS],
    }


def get_target_urls(base_url: str) -> list[str]:
    """Flat, tier-ordered view of get_tiered_target_urls() (tier1, tier2,
    tier3 - homepage last). Nothing is crawled or followed; this is the
    entire search space."""
    tiers = get_tiered_target_urls(base_url)
    return tiers["tier1"] + tiers["tier2"] + tiers["tier3"]


def _classify_fetch_error(exc: Exception) -> str:
    """Map a raw Playwright/Chromium exception to a short, human-readable
    failure reason, for the debug_log surfaced when a check comes back
    "unreachable" (see run_check())."""
    message = str(exc)
    lower = message.lower()
    if "timeout" in lower:
        return "timeout"
    if "err_name_not_resolved" in lower or "getaddrinfo" in lower or "dns" in lower:
        return "DNS failure"
    if "err_connection_refused" in lower or "connection refused" in lower:
        return "connection refused"
    if "err_connection_reset" in lower:
        return "connection reset"
    if "err_connection_closed" in lower:
        return "connection closed"
    if "err_cert" in lower or "certificate" in lower:
        return "TLS/certificate error"
    if "net::" in message:
        # Fall back to Chromium's own net:: error code, e.g. "ERR_ABORTED".
        return message.split("net::", 1)[1].split()[0].strip(" .")
    stripped = message.strip()
    return (stripped[:120] + "...") if len(stripped) > 120 else (stripped or "unknown error")


def fetch_page_with_status(url: str) -> tuple[str | None, str, str | None]:
    """Load `url` with a headless browser.

    Returns (html, status, reason) where status is one of:
      "ok"          - page loaded successfully, html is the rendered content
      "404"         - page does not exist; expected/normal, not retried
      "unreachable" - couldn't connect/load at all, retried once first
    reason is None for "ok", and a short human-readable failure description
    otherwise (e.g. "timeout", "DNS failure", "connection refused",
    "Not Found (404)") - see _classify_fetch_error().
    """
    attempts = FETCH_RETRIES + 1
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(
                    user_agent=_USER_AGENT,
                    viewport=_VIEWPORT,
                    locale="en-US",
                    extra_http_headers=_EXTRA_HEADERS,
                )
                try:
                    page = context.new_page()
                    response = page.goto(url, timeout=FETCH_TIMEOUT_MS, wait_until="networkidle")

                    # Best-effort: give client-rendered content a chance to
                    # paint before we grab the DOM. Don't fail the whole
                    # fetch if nothing matches within budget - just log it
                    # and fall through to whatever's there.
                    try:
                        page.wait_for_selector("body", timeout=FETCH_TIMEOUT_MS)
                    except Exception as wait_exc:  # noqa: BLE001
                        print(f"fetch_page: wait_for_selector timed out for {url!r}: {wait_exc}")

                    status_code = response.status if response is not None else None
                    print(f"fetch_page: {url} -> status={status_code}, title={page.title()!r}")

                    if status_code == 404:
                        return None, "404", "Not Found (404)"
                    return page.content(), "ok", None
                finally:
                    context.close()
                    browser.close()
        except Exception as exc:  # noqa: BLE001 - broad by design, we retry/return None
            last_error = exc
            continue

    reason = _classify_fetch_error(last_error) if last_error is not None else "unknown error"
    if last_error is not None:
        print(f"fetch_page: failed to load {url!r}: {last_error}")
    return None, "unreachable", reason


def fetch_page(url: str) -> str | None:
    """Load `url` and return the fully rendered HTML.

    Returns None if the page 404s or is unreachable. See
    fetch_page_with_status() for a version that distinguishes the two (and
    surfaces WHY it failed).
    """
    html, _status, _reason = fetch_page_with_status(url)
    return html


def extract_text(html: str, source_url: str = "") -> dict:
    """Pull title, meta description, footer text, copyright notice, and body
    text from `html`. `source_url` is carried through so callers know which
    sub-URL the text came from.

    body_text is always the FULL page text, never truncated/snippeted -
    uniformly for every page type (homepage, about, press, legal/privacy/
    terms, ...), not just long legal pages. There is no length cap or
    snippet/head-only extraction anywhere in this function; the acquisition
    patterns in check_acquirer_match() need the complete text since a
    disclosure can be buried anywhere in the document, on any page.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    footer_text = ""
    footer_tag = soup.find("footer")
    if footer_tag:
        footer_text = footer_tag.get_text(separator="\n", strip=True)

    # Drop non-visible elements before pulling general body text.
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    body_tag = soup.find("body")
    # Newline (not space) separator: distinct DOM blocks (a headline, a
    # date stamp, a "Download PDF" button, ...) shouldn't get concatenated
    # into one run-on clause for the regexes below to accidentally span
    # across.
    body_text = (body_tag or soup).get_text(separator="\n", strip=True)

    copyright_match = _COPYRIGHT_PATTERN.search(footer_text) or _COPYRIGHT_PATTERN.search(body_text)
    copyright_text = copyright_match.group(0).strip() if copyright_match else ""

    return {
        "title": title,
        "description": description,
        "footer": footer_text,
        "body_text": body_text,
        "copyright": copyright_text,
        "source_url": source_url,
    }


def _clean_company_name(raw: str) -> str | None:
    raw = raw.strip()
    cutoff = _TRAILING_CUTOFF.search(raw)
    name = raw[: cutoff.start()] if cutoff else raw
    name = name.strip(" \t\"'()[]{}")
    return name or None


def _names_match(a: str, b: str) -> bool:
    """Loose equality: ignores case and treats one name containing the other
    (e.g. 'Acme' vs 'Acme Inc') as the same company."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    return a in b or b in a


def _is_denylisted(candidate: str) -> bool:
    lower = candidate.lower().strip()
    if not lower:
        return True
    return any(term in lower or lower in term for term in _DENYLIST)


def _is_generic_term(candidate: str) -> bool:
    """True only if the ENTIRE candidate is a generic business/industry
    word (e.g. "SaaS", "Cloud") - an exact match, not substring containment,
    so a legitimate name like "Global Payments Inc" or a short acronym like
    "VWO" is never rejected just for containing/resembling a common word."""
    return candidate.strip().lower() in _GENERIC_TERM_DENYLIST


def _words_around(text: str, start: int, end: int, window_words: int) -> str:
    before_words = text[:start].split()[-window_words:]
    after_words = text[end:].split()[:window_words]
    return " ".join(before_words + after_words).lower()


def _has_hypothetical_language(text: str, start: int, end: int, window_words: int = 15) -> bool:
    """True if a hypothetical/conditional marker ("may be", "in the event
    of", ...) appears in the ~window_words words around the match - i.e.
    standard "if we are ever acquired" privacy-policy boilerplate, not
    evidence of an actual acquisition."""
    window = _words_around(text, start, end, window_words)
    return any(marker in window for marker in _HYPOTHETICAL_MARKERS)


def _has_self_description_nearby(text: str, start: int, end: int, window_words: int = 15) -> bool:
    """For the 'an? X company' pattern: True if the site is describing
    itself ('we are ... company', 'is an independent ... company') rather
    than naming an acquirer."""
    window = _words_around(text, start, end, window_words)
    return any(marker in window for marker in _SELF_DESCRIPTION_MARKERS)


def _full_line(text: str, start: int, end: int) -> str:
    """The complete line (DOM block/sentence) containing the match - text is
    \\n-separated per DOM block, see extract_text()."""
    line_start = text.rfind("\n", 0, start)
    line_start = 0 if line_start == -1 else line_start + 1
    line_end = text.find("\n", end)
    line_end = len(text) if line_end == -1 else line_end
    return text[line_start:line_end].strip()


def _passes_validation(candidate: str, company_name: str) -> bool:
    """Reject a candidate if it's denylisted noise, a generic business/
    industry term rather than a real name, the company's own name, outside
    the 3-50 char shape a real company name should have, or doesn't start
    with a capital letter."""
    if _is_denylisted(candidate):
        return False
    if _is_generic_term(candidate):
        return False
    if _names_match(candidate, company_name):
        return False
    if not (3 <= len(candidate) <= 50):
        return False
    if not candidate[0].isupper():
        return False
    return True


def check_acquirer_match(extracted: dict, company_name: str) -> list[dict]:
    """Scan extracted text fields for STRICT, factual acquisition phrasing
    only (see _STRICT_ACQUIRER_PATTERNS) - deliberately narrower than a
    general "sounds acquisition-related" scan, because that looser approach
    previously flagged "Registration Data" (WHOIS/registrar boilerplate in
    a privacy policy) as an acquirer for Whimsical. It is better to return
    no evidence than to fabricate a match.

    Every candidate must clear three gates in order:
      1. Not wrapped in hypothetical/conditional language ("may be", "in
         the event of", ...) - "if we are ever acquired" is not evidence.
      2. (a_x_company pattern only) Not the site describing its own
         independence ("we are a privately held company").
      3. Not denylisted, not the company's own name, and shaped like a
         real name (3-50 chars, starts with a capital letter).

    Returns a list of dicts - one per surviving, deduplicated candidate -
    each with {candidate_name, matched_pattern, confidence, source_url,
    matched_sentence}. confidence ("high" | "medium") reflects how
    unambiguous the matched phrasing is - see _PATTERN_CONFIDENCE. Does NOT
    stop at the first match: if the page's text yields more than one
    distinct candidate name, all of them are returned so the caller can
    surface the ambiguity rather than silently picking one.
    """
    results: list[dict] = []
    seen: set[str] = set()
    source_url = extracted.get("source_url", "")

    for field in ("title", "description", "footer", "body_text"):
        text = extracted.get(field) or ""
        for pattern_name, pattern in _STRICT_ACQUIRER_PATTERNS:
            for match in pattern.finditer(text):
                if _has_hypothetical_language(text, match.start(), match.end()):
                    continue
                if pattern_name == "a_x_company" and _has_self_description_nearby(
                    text, match.start(), match.end()
                ):
                    continue

                candidate = _clean_company_name(match.group(1))
                if not candidate or not _passes_validation(candidate, company_name):
                    continue

                dedupe_key = f"{candidate.lower()}|{source_url}"
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)

                results.append(
                    {
                        "candidate_name": candidate,
                        "matched_pattern": pattern_name,
                        "confidence": _PATTERN_CONFIDENCE[pattern_name],
                        "source_url": source_url,
                        "matched_sentence": _full_line(text, match.start(), match.end()),
                    }
                )

    return results


def _is_listing_url(url: str) -> bool:
    path = urlparse(url).path.rstrip("/") or "/"
    return path in _LISTING_PATHS


def _extract_acquisition_leaning_links(html: str, base_url: str, limit: int = _MAX_LISTING_SUBPAGES) -> list[str]:
    """Pull same-domain links from a press/news listing page whose slug
    looks acquisition-related (e.g. '.../permira-completes-acquisition-of-
    squarespace'). Bounded and keyword-filtered - one level deep off a known
    listing page, not general crawling.
    """
    soup = BeautifulSoup(html, "html.parser")
    parsed_base = urlparse(base_url)
    seen: set[str] = set()
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)
        if parsed.netloc and parsed.netloc != parsed_base.netloc:
            continue
        if not any(kw in parsed.path.lower() for kw in _ACQUISITION_SLUG_KEYWORDS):
            continue

        clean = parsed._replace(query="", fragment="").geturl()
        if clean in seen:
            continue
        seen.add(clean)
        links.append(clean)
        if len(links) >= limit:
            break

    return links


def _scan_tier(urls: list[str], company_name: str) -> tuple[list[dict], bool, list[dict]]:
    """Fetch and check each URL in a tier. Returns (evidence, any_reachable,
    debug_log).

    Collects evidence from every reachable URL in the tier (does not stop
    at the first candidate - a single page can legitimately yield more than
    one distinct candidate name, and that ambiguity should surface, not be
    silently discarded). For listing pages (e.g. /press-releases), also
    fetches/checks a bounded, keyword-filtered set of individual article
    links found on that page - see _extract_acquisition_leaning_links().

    debug_log records every URL attempted in this tier with its outcome
    (status + failure reason for anything that wasn't "ok"), so a final
    "unreachable" result can show exactly what was tried and why each
    attempt failed, instead of a single opaque status.
    """
    any_reachable = False
    evidence: list[dict] = []
    debug_log: list[dict] = []

    for url in urls:
        html, status, reason = fetch_page_with_status(url)
        debug_log.append({"url": url, "status": status, "reason": reason})
        if status != "ok" or html is None:
            continue
        any_reachable = True

        extracted = extract_text(html, source_url=url)
        evidence.extend(check_acquirer_match(extracted, company_name))

        if not _is_listing_url(url):
            continue

        for sub_url in _extract_acquisition_leaning_links(html, url):
            sub_html, sub_status, sub_reason = fetch_page_with_status(sub_url)
            debug_log.append({"url": sub_url, "status": sub_status, "reason": sub_reason})
            if sub_status != "ok" or sub_html is None:
                continue
            any_reachable = True

            sub_extracted = extract_text(sub_html, source_url=sub_url)
            evidence.extend(check_acquirer_match(sub_extracted, company_name))

    return evidence, any_reachable, debug_log


def run_check(company_url: str, company_name: str) -> dict:
    """Fetch, extract, and match a company's site for acquisition signals,
    scanning candidate sub-URLs tier by tier (high-signal press/about pages
    and legal pages first; the noisier homepage last, only if the other
    tiers found no evidence).

    Returns a structured result:
      {
        "company_name": str,
        "is_acquired": bool,       # True only if >=1 candidate survived validation
        "confidence": "high" | "medium" | "unconfirmed",
        "likely_acquirer": str | None,  # set only when exactly one distinct
                                         # candidate name was found; None if
                                         # zero, or if multiple conflicting
                                         # names were found (ambiguous - left
                                         # for human review, not guessed)
        "evidence": [{candidate_name, source_url, matched_sentence}, ...],
        "status": "checked" | "unreachable",
        "debug_log": [{url, status, reason}, ...],  # every URL attempted;
                                                      # populated whenever
                                                      # status="unreachable",
                                                      # [] otherwise
      }

    confidence reflects the strength of the matched phrasing (see
    _PATTERN_CONFIDENCE): "high" for an unambiguous claim like "was
    acquired by" / "is a subsidiary of", "medium" for softer language like
    "backed by" / "a X brand". When a single candidate matched via more
    than one pattern, the strongest tier wins. Multiple DIFFERENT candidate
    names always downgrades to "unconfirmed" regardless of pattern strength
    - that ambiguity is left for human review, not guessed.

    Never infers acquisition from missing data: if no candidate URL was
    reachable at all, status is "unreachable" and is_acquired is False -
    not "checked" with a false negative. debug_log then shows exactly which
    URLs were tried and why each one failed (timeout / 404 / DNS failure /
    connection refused / ...).
    """
    tiers = get_tiered_target_urls(company_url)

    any_reachable = False
    evidence: list[dict] = []
    debug_log: list[dict] = []

    for tier_name in ("tier1", "tier2", "tier3"):
        tier_evidence, tier_reachable, tier_debug_log = _scan_tier(tiers[tier_name], company_name)
        any_reachable = any_reachable or tier_reachable
        evidence.extend(tier_evidence)
        debug_log.extend(tier_debug_log)
        if evidence:
            break  # stop escalating tiers once we have evidence to report

    if not any_reachable:
        return {
            "company_name": company_name,
            "is_acquired": False,
            "confidence": "unconfirmed",
            "likely_acquirer": None,
            "evidence": [],
            "status": "unreachable",
            "debug_log": debug_log,
        }

    if not evidence:
        return {
            "company_name": company_name,
            "is_acquired": False,
            "confidence": "unconfirmed",
            "likely_acquirer": None,
            "evidence": [],
            "status": "checked",
            "debug_log": [],
        }

    distinct_names = {e["candidate_name"] for e in evidence}
    single_candidate = len(distinct_names) == 1

    if single_candidate:
        best_confidence = max(
            (e["confidence"] for e in evidence),
            key=lambda c: _CONFIDENCE_RANK[c],
        )
    else:
        best_confidence = "unconfirmed"

    return {
        "company_name": company_name,
        "is_acquired": True,
        "confidence": best_confidence,
        "likely_acquirer": next(iter(distinct_names)) if single_candidate else None,
        "evidence": [
            {
                "candidate_name": e["candidate_name"],
                "source_url": e["source_url"],
                "matched_sentence": e["matched_sentence"],
            }
            for e in evidence
        ],
        "status": "checked",
        "debug_log": [],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (2, 3):
        print("Usage: python check_text_signals.py <url> [company_name]")
        sys.exit(1)

    url = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) == 3 else derive_company_name(url)
    import json

    print(json.dumps(run_check(url, name), indent=2))
