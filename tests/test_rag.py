"""Tests for the RAG Layer module."""

from __future__ import annotations

import pytest

from utils.parsing import classify_domain_tier, extract_domain, url_hash


# ─────────────────────────────────────────────────────────────────────────────
#  Utility tests (these don't need mocking)
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractDomain:
    def test_simple_url(self):
        assert extract_domain("https://arxiv.org/abs/2301.00001") == "arxiv.org"

    def test_www_prefix(self):
        assert extract_domain("https://www.nature.com/articles/123") == "nature.com"

    def test_subdomain(self):
        assert (
            extract_domain("https://pubmed.ncbi.nlm.nih.gov/12345")
            == "pubmed.ncbi.nlm.nih.gov"
        )

    def test_invalid_url(self):
        assert extract_domain("not-a-url") == ""


class TestClassifyDomainTier:
    def test_tier1(self):
        assert classify_domain_tier("arxiv.org") == "tier_1"
        assert classify_domain_tier("pubmed.ncbi.nlm.nih.gov") == "tier_1"

    def test_tier2(self):
        assert classify_domain_tier("reuters.com") == "tier_2"

    def test_tier3(self):
        assert classify_domain_tier("medium.com") == "tier_3"

    def test_unknown(self):
        assert classify_domain_tier("randomsite.xyz") == "unknown"


class TestUrlHash:
    def test_deterministic(self):
        h1 = url_hash("https://example.com/page")
        h2 = url_hash("https://example.com/page")
        assert h1 == h2

    def test_length(self):
        assert len(url_hash("https://example.com")) == 12

    def test_different_urls_different_hashes(self):
        assert url_hash("https://a.com") != url_hash("https://b.com")
