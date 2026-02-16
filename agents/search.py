"""Active Scholar — Search Agent nodes.

Nodes
-----
- ``search_generate_queries`` – Uses the LLM to produce diverse search queries.
- ``search_execute``          – Runs queries via Tavily (primary) / Serper (fallback).
- ``search_evaluate_credibility`` – Scores each source for credibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from config import settings
from state import ResearchState, SourceMetadata
from utils.llm import llm_call
from utils.parsing import classify_domain_tier, extract_domain

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Query Generation
# ═══════════════════════════════════════════════════════════════════════════════

_QUERY_GEN_TEMPLATE = (_PROMPT_DIR / "query_gen.txt").read_text(encoding="utf-8")


async def search_generate_queries(state: ResearchState) -> dict:
    """Generate diverse search queries from the research topic."""
    scope_map = {"narrow": 3, "moderate": 5, "broad": 8}
    n = scope_map.get(state.get("scope", "moderate"), 5)

    prompt = _QUERY_GEN_TEMPLATE.format(
        n=n,
        topic=state["topic"],
        scope=state.get("scope", "moderate"),
        constraints=", ".join(state.get("constraints", [])),
        previous_queries=json.dumps(state.get("search_queries", [])),
        follow_up_queries=json.dumps(state.get("follow_up_queries", [])),
    )

    raw = await llm_call(prompt)

    # Robust JSON extraction — handle markdown-fenced responses
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        new_queries: list[str] = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse query-gen response: %s", text[:200])
        new_queries = [state["topic"]]  # safe fallback

    return {
        "search_queries": state.get("search_queries", []) + new_queries,
        "current_search_round": state.get("current_search_round", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Search Execution
# ═══════════════════════════════════════════════════════════════════════════════


async def _tavily_search(query: str) -> list[dict]:
    """Execute a search via the Tavily Python SDK."""
    from tavily import TavilyClient  # lazy so missing dep doesn't block import

    client = TavilyClient(api_key=settings.tavily_api_key)
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=settings.max_results_per_query,
            include_raw_content=False,
        )
        return resp.get("results", [])
    except Exception as exc:
        logger.warning("Tavily search failed for '%s': %s", query, exc)
        return []


async def _serper_fallback(query: str) -> list[dict]:
    """Execute a search via the Serper.dev API (fallback)."""
    if not settings.serper_api_key:
        return []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": settings.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": settings.max_results_per_query},
            )
            resp.raise_for_status()
            data = resp.json()
        # Normalise to Tavily-like shape
        return [
            {
                "url": r.get("link", ""),
                "title": r.get("title", ""),
                "content": r.get("snippet", ""),
            }
            for r in data.get("organic", [])
        ]
    except Exception as exc:
        logger.warning("Serper fallback failed for '%s': %s", query, exc)
        return []


async def search_execute(state: ResearchState) -> dict:
    """Run the latest batch of queries and collect new sources."""
    queries = state.get("search_queries", [])
    # Only run the latest batch (queries added in this round)
    scope_map = {"narrow": 3, "moderate": 5, "broad": 8}
    batch_size = scope_map.get(state.get("scope", "moderate"), 5)
    latest_queries = queries[-batch_size:]

    explored: set[str] = set(state.get("explored_urls", set()))
    new_sources: list[SourceMetadata] = []
    exhausted = True

    for query in latest_queries:
        results = await _tavily_search(query)
        if not results:
            results = await _serper_fallback(query)

        for r in results:
            url = r.get("url", "")
            if not url or url in explored:
                continue
            exhausted = False
            explored.add(url)
            new_sources.append(
                SourceMetadata(
                    url=url,
                    title=r.get("title", ""),
                    domain=extract_domain(url),
                    discovered_at=datetime.now(timezone.utc).isoformat(),
                    credibility_score=0.0,
                    source_type="unknown",
                    is_paywalled=False,
                    raw_snippet=(r.get("content", "") or "")[:500],
                )
            )

    return {
        "search_results": state.get("search_results", []) + new_sources,
        "explored_urls": explored,
        "search_exhausted": exhausted,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Credibility Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

_CRED_TEMPLATE = (_PROMPT_DIR / "credibility.txt").read_text(encoding="utf-8")


async def search_evaluate_credibility(state: ResearchState) -> dict:
    """Score ALL unevaluated sources for credibility in a single batched LLM call."""
    all_sources: list[SourceMetadata] = list(state.get("search_results", []))

    # Separate already-evaluated from needing evaluation
    already_done: list[SourceMetadata] = []
    needs_eval: list[SourceMetadata] = []
    for source in all_sources:
        if source.credibility_score > 0:
            already_done.append(source)
        else:
            needs_eval.append(source)

    if not needs_eval:
        return {"search_results": already_done}

    # Build a single sources block for the batched prompt
    source_entries = []
    for source in needs_eval:
        tier = classify_domain_tier(source.domain)
        source_entries.append(
            f"- URL: {source.url}\n"
            f"  Title: {source.title}\n"
            f"  Snippet: {source.raw_snippet[:300]}\n"
            f"  Domain tier hint: {tier}"
        )
    sources_block = "\n".join(source_entries)

    prompt = _CRED_TEMPLATE.format(sources_block=sources_block)

    try:
        raw = await llm_call(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        eval_list = json.loads(text)

        # Build a lookup by URL
        eval_lookup: dict[str, dict] = {}
        if isinstance(eval_list, list):
            for item in eval_list:
                url = item.get("url", "")
                if url:
                    eval_lookup[url] = item

        for source in needs_eval:
            match = eval_lookup.get(source.url)
            if match:
                source.credibility_score = float(match.get("credibility_score", 0.5))
                source.source_type = match.get("source_type", "unknown")
                source.is_paywalled = bool(match.get("is_paywalled", False))
            else:
                source.credibility_score = 0.5  # neutral default

    except Exception as exc:
        logger.warning("Batched credibility eval failed: %s", exc)
        for source in needs_eval:
            source.credibility_score = 0.5  # neutral default

    updated = already_done + needs_eval

    # Filter below minimum threshold
    filtered = [
        s for s in updated if s.credibility_score >= settings.min_credibility_score
    ]
    return {"search_results": filtered}
