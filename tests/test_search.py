"""Tests for the Search Agent module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from state import ResearchState


@pytest.fixture
def empty_state() -> ResearchState:
    return ResearchState(
        topic="Impact of microplastics on marine ecosystems",
        scope="moderate",
        constraints=["published after 2022"],
        max_search_rounds=3,
        search_queries=[],
        search_results=[],
        explored_urls=set(),
        current_search_round=0,
        search_exhausted=False,
        ingested_chunks=[],
        retrieval_results=[],
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
async def test_generate_queries_returns_list(empty_state):
    """generate_queries should return a list of query strings."""
    mock_response = json.dumps([
        "microplastics marine impact 2023",
        "ocean pollution plastic research",
        "microplastic toxicity fish studies",
        "counterarguments microplastic harm oceans",
        "recent microplastics research 2024",
    ])
    with patch("agents.search.llm_call", new_callable=AsyncMock, return_value=mock_response):
        from agents.search import search_generate_queries

        result = await search_generate_queries(empty_state)

    assert "search_queries" in result
    assert len(result["search_queries"]) == 5
    assert result["current_search_round"] == 1


@pytest.mark.asyncio
async def test_generate_queries_handles_markdown_fenced_json(empty_state):
    """Query generation should handle LLM responses wrapped in code fences."""
    fenced = '```json\n["query one", "query two"]\n```'
    with patch("agents.search.llm_call", new_callable=AsyncMock, return_value=fenced):
        from agents.search import search_generate_queries

        result = await search_generate_queries(empty_state)

    assert len(result["search_queries"]) == 2


@pytest.mark.asyncio
async def test_generate_queries_fallback_on_bad_json(empty_state):
    """If the LLM returns unparsable JSON, fall back to the raw topic."""
    with patch("agents.search.llm_call", new_callable=AsyncMock, return_value="NOT JSON"):
        from agents.search import search_generate_queries

        result = await search_generate_queries(empty_state)

    # Should still return at least one query (the topic itself)
    assert len(result["search_queries"]) >= 1
    assert empty_state["topic"] in result["search_queries"]
