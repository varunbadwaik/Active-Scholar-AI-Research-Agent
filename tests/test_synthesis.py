"""Tests for the Synthesis Agent module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from state import Claim, DocumentChunk, ResearchState, SourceMetadata


@pytest.fixture
def state_with_retrieval() -> ResearchState:
    """State that already has retrieved chunks ready for synthesis."""
    return ResearchState(
        topic="AI safety alignment techniques",
        scope="moderate",
        constraints=[],
        max_search_rounds=3,
        search_queries=["AI alignment safety"],
        search_results=[
            SourceMetadata(
                url="https://arxiv.org/abs/2301.00001",
                title="RLHF Safety Survey",
                domain="arxiv.org",
                discovered_at="2026-02-16T00:00:00Z",
                credibility_score=0.95,
                source_type="academic",
            ),
            SourceMetadata(
                url="https://example-blog.com/ai-safety",
                title="My Thoughts on AI Safety",
                domain="example-blog.com",
                discovered_at="2026-02-16T00:00:00Z",
                credibility_score=0.4,
                source_type="blog",
            ),
        ],
        explored_urls=set(),
        current_search_round=1,
        search_exhausted=False,
        ingested_chunks=[],
        retrieval_results=[
            DocumentChunk(
                chunk_id="abc_0",
                source_url="https://arxiv.org/abs/2301.00001",
                text="RLHF has shown to reduce harmful outputs by 70%.",
                metadata={
                    "source_title": "RLHF Safety Survey",
                    "source_url": "https://arxiv.org/abs/2301.00001",
                    "credibility_score": 0.95,
                },
            ),
            DocumentChunk(
                chunk_id="def_0",
                source_url="https://example-blog.com/ai-safety",
                text="RLHF only reduces harmful outputs by 30%, and often fails.",
                metadata={
                    "source_title": "My Thoughts on AI Safety",
                    "source_url": "https://example-blog.com/ai-safety",
                    "credibility_score": 0.4,
                },
            ),
        ],
        claims=[],
        contradictions=[],
        evidence_quality={},
        synthesis_complete=False,
        needs_more_info=False,
        follow_up_queries=[],
        report=None,
        report_metadata=None,
    )


# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_extracts_claims(state_with_retrieval):
    """synthesis_analyze should extract claims from passages."""
    mock_claims = json.dumps([
        {
            "claim_text": "RLHF reduces harmful outputs by 70%",
            "supporting_sources": ["https://arxiv.org/abs/2301.00001"],
            "confidence": 0.9,
        },
    ])

    # Mock both llm_call and verify_claim_grounding
    with (
        patch("agents.synthesis.llm_call", new_callable=AsyncMock, return_value=mock_claims),
        patch("agents.synthesis.verify_claim_grounding", new_callable=AsyncMock, return_value=True),
        patch("agents.synthesis.get_collection", return_value=MagicMock()),
    ):
        from agents.synthesis import synthesis_analyze

        result = await synthesis_analyze(state_with_retrieval)

    assert "claims" in result
    assert len(result["claims"]) == 1
    assert result["claims"][0].confidence == 0.9


@pytest.mark.asyncio
async def test_detect_conflicts(state_with_retrieval):
    """synthesis_detect_conflicts should surface contradictions."""
    # Pre-populate claims
    state_with_retrieval["claims"] = [
        Claim(
            claim_text="RLHF reduces harmful outputs by 70%",
            supporting_sources=["https://arxiv.org/abs/2301.00001"],
            confidence=0.9,
        ),
        Claim(
            claim_text="RLHF only reduces harmful outputs by 30%",
            supporting_sources=["https://example-blog.com/ai-safety"],
            confidence=0.5,
        ),
    ]

    mock_conflicts = json.dumps([
        {
            "claim_a_index": 0,
            "claim_b_index": 1,
            "conflict_type": "statistical_discrepancy",
            "explanation": "Contradictory percentages for RLHF effectiveness",
            "resolution_suggestion": "Check original study methodology",
            "which_is_stronger": "a",
            "reasoning": "Source A is a peer-reviewed survey with higher credibility",
        }
    ])

    with patch("agents.synthesis.llm_call", new_callable=AsyncMock, return_value=mock_conflicts):
        from agents.synthesis import synthesis_detect_conflicts

        result = await synthesis_detect_conflicts(state_with_retrieval)

    assert len(result["contradictions"]) == 1
    assert result["contradictions"][0]["conflict_type"] == "statistical_discrepancy"
    assert result["needs_more_info"] is False  # resolved (not inconclusive)


@pytest.mark.asyncio
async def test_report_generation(state_with_retrieval):
    """synthesis_generate_report should produce markdown and metadata."""
    state_with_retrieval["claims"] = [
        Claim(
            claim_text="RLHF is effective",
            supporting_sources=["https://arxiv.org/abs/2301.00001"],
            confidence=0.9,
        )
    ]
    state_with_retrieval["contradictions"] = []
    state_with_retrieval["evidence_quality"] = {}

    mock_report = "# Research Report\n\n## Executive Summary\nFindings here."
    with patch("agents.synthesis.llm_call", new_callable=AsyncMock, return_value=mock_report):
        from agents.synthesis import synthesis_generate_report

        result = await synthesis_generate_report(state_with_retrieval)

    assert result["synthesis_complete"] is True
    assert "# Research Report" in result["report"]
    assert result["report_metadata"]["overall_confidence"] == "HIGH"
