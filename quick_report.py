"""Active Scholar — Quick Report Generator.

A fast, reliable alternative to the full LangGraph pipeline.
Generates a research report in ~30 seconds using:
  1. One LLM call to generate search queries
  2. Tavily search (uses returned snippets — no URL fetching)
  3. One LLM call to generate the full report from snippets

Falls back to a deterministic report if the LLM fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from config import settings
from utils.llm import _get_client, _rate_limit, llm_call

logger = logging.getLogger(__name__)


# ── Quick report generation ──────────────────────────────────────────────────


async def generate_quick_report(
    topic: str,
    scope: str = "moderate",
    max_results: int = 5,
) -> dict:
    """Generate a research report quickly using search snippets + one LLM call.

    Returns a dict with keys: report, report_metadata, sources, claims, contradictions.
    """
    logger.info("Quick report started — topic: %s", topic)
    t0 = asyncio.get_event_loop().time()

    # ── Step 1: Search ───────────────────────────────────────────────────────
    sources = await _search_tavily(topic, max_results=max_results)
    if not sources:
        logger.warning("No search results — generating report from topic alone")

    logger.info("Found %d sources in %.1fs", len(sources), asyncio.get_event_loop().time() - t0)

    # ── Step 2: Generate report ──────────────────────────────────────────────
    snippets_text = _format_snippets(sources)
    report_md = await _generate_report_from_snippets(topic, scope, snippets_text)

    elapsed = asyncio.get_event_loop().time() - t0
    logger.info("Quick report completed in %.1fs (%d chars)", elapsed, len(report_md))

    # ── Build result ─────────────────────────────────────────────────────────
    return {
        "report": report_md,
        "report_markdown": report_md,
        "report_metadata": {
            "topic": topic,
            "scope": scope,
            "total_sources_discovered": len(sources),
            "total_sources_used": len(sources),
            "total_claims_extracted": 0,
            "total_contradictions_found": 0,
            "search_rounds_completed": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_confidence": "MODERATE",
            "warnings": [],
            "mode": "quick",
            "generation_time_seconds": round(elapsed, 1),
        },
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "domain": s.get("domain", ""),
                "credibility_score": 0.5,
                "source_type": "unknown",
            }
            for s in sources
        ],
        "claims": [],
        "contradictions": [],
    }


# ── Tavily search ────────────────────────────────────────────────────────────


async def _search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """Search using Tavily and return results with snippets."""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        resp = await asyncio.to_thread(
            client.search,
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
        results = resp.get("results", [])
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": (r.get("content", "") or "")[:1000],
                "domain": _extract_domain(r.get("url", "")),
            }
            for r in results
            if r.get("url")
        ]
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        return []


def _extract_domain(url: str) -> str:
    """Extract domain from URL."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return "unknown"


# ── Snippet formatting ───────────────────────────────────────────────────────


def _format_snippets(sources: list[dict]) -> str:
    """Format search results into a text block for the LLM."""
    if not sources:
        return "No sources found."

    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(
            f"[Source {i}] {s['title']}\n"
            f"URL: {s['url']}\n"
            f"Domain: {s['domain']}\n"
            f"{s['snippet']}\n"
        )
    return "\n---\n".join(parts)


# ── Report generation ────────────────────────────────────────────────────────

_QUICK_REPORT_PROMPT = """You are a research analyst. Write a comprehensive, well-structured research report on the following topic using ONLY the provided source snippets.

Topic: {topic}
Scope: {scope}

Source Snippets:
{snippets}

Write the report in Markdown with these sections:

# Research Report: {topic}

## Executive Summary
2-3 paragraph overview of key findings.

## Key Findings
Numbered list of the most important facts, each citing its source as [Source Title](URL).

## Detailed Analysis
In-depth discussion organized by theme/subtopic. Every claim must cite a source.

## Source Evaluation
Brief assessment of source quality and diversity.

## Gaps & Limitations
What could NOT be determined from available sources.

## Conclusion
Synthesized position with confidence assessment.

## References
Numbered list of all sources used: [N] Title. URL.

Rules:
- EVERY factual statement must cite its source with a link.
- Be thorough but concise.
- If sources conflict, note the contradiction explicitly.
- Do NOT invent information not in the snippets.
"""


async def _generate_report_from_snippets(
    topic: str, scope: str, snippets: str
) -> str:
    """Generate the report with a single LLM call + hard timeout."""
    prompt = _QUICK_REPORT_PROMPT.format(
        topic=topic, scope=scope, snippets=snippets
    )

    try:
        # Single LLM call with 90-second timeout
        await _rate_limit()
        client = _get_client()
        model_name = settings.primary_model

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
            ),
            timeout=90.0,
        )
        report = response.text or ""
        if report.strip():
            return report
        raise ValueError("Empty LLM response")

    except Exception as exc:
        logger.error("LLM report generation failed: %s", exc)
        logger.info("Using fallback deterministic report")
        return _fallback_report(topic, snippets)


def _fallback_report(topic: str, snippets: str) -> str:
    """Generate a deterministic report when the LLM is unavailable."""
    return f"""# Research Report: {topic}

> **Note:** This report was generated automatically from search snippets because the AI synthesis engine was unavailable.

## Source Snippets

{snippets}

---
*Report generated at {datetime.now(timezone.utc).isoformat()} using Active Scholar Quick Mode.*
"""
