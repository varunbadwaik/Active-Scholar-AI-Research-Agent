"""Active Scholar — Document fetching, text extraction, and misc utilities."""

from __future__ import annotations

import hashlib
import logging
from urllib.parse import urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)

# ── Domain helpers ───────────────────────────────────────────────────────────


def extract_domain(url: str) -> str:
    """Return the bare domain (e.g. ``'arxiv.org'``) from a full URL."""
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "unknown"


def url_hash(url: str) -> str:
    """Short, deterministic hash of a URL for use as a chunk-ID prefix."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


# ── Domain-tier classification ───────────────────────────────────────────────

DOMAIN_TIERS: dict[str, list[str]] = {
    "tier_1": [
        ".gov",
        ".edu",
        "nature.com",
        "science.org",
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "ieee.org",
        "springer.com",
        "wiley.com",
        "acm.org",
    ],
    "tier_2": [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "washingtonpost.com",
        "economist.com",
    ],
    "tier_3": [
        "medium.com",
        "substack.com",
        "wordpress.com",
        "blogspot.com",
        "reddit.com",
    ],
}


def classify_domain_tier(domain: str) -> str:
    """Classify a domain into tier_1 / tier_2 / tier_3 / unknown."""
    for tier, patterns in DOMAIN_TIERS.items():
        for pattern in patterns:
            if domain.endswith(pattern):
                return tier
    return "unknown"


# ── Full-text fetching ───────────────────────────────────────────────────────

_HTTP_TIMEOUT = 15.0  # seconds


async def fetch_full_text(url: str) -> str:
    """Download a URL and extract the main textual content.

    Uses *trafilatura* for main-content extraction with an *httpx* fallback to
    raw HTML scraping via BeautifulSoup.  Returns an empty string if the
    content cannot be fetched or extracted.
    """
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": "ActiveScholar/1.0 (research-agent)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""

    # Primary extraction via trafilatura
    text = trafilatura.extract(html, include_comments=False, include_tables=True)
    if text:
        return text

    # Fallback: simple BeautifulSoup extraction
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as exc:
        logger.warning("BeautifulSoup fallback failed for %s: %s", url, exc)
        return ""
