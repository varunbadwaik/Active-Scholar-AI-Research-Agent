"""Active Scholar — Synthesis Agent nodes.

Nodes
-----
- ``synthesis_analyze``           – Extracts discrete claims from retrieved chunks.
- ``synthesis_detect_conflicts``  – Finds contradictions between claims.
- ``synthesis_generate_report``   – Produces the final structured Markdown report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from state import Claim, ResearchState
from utils.llm import llm_call
from utils.verification import verify_claim_grounding

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# ═══════════════════════════════════════════════════════════════════════════════
#  1. Claim Extraction
# ═══════════════════════════════════════════════════════════════════════════════

_CLAIM_TEMPLATE = (_PROMPT_DIR / "claim_extraction.txt").read_text(encoding="utf-8")


async def synthesis_analyze(state: ResearchState) -> dict:
    """Extract atomic, verifiable claims from retrieved passages."""
    retrieval = state.get("retrieval_results", [])
    if not retrieval:
        logger.warning("No retrieval results — cannot extract claims")
        return {"claims": state.get("claims", [])}

    # Format passages with attribution
    passages = "\n\n".join(
        f"[Source: {c.metadata.get('source_title', 'Unknown')} | "
        f"{c.source_url} | "
        f"Credibility: {c.metadata.get('credibility_score', '?')}]\n{c.text}"
        for c in retrieval
    )

    prompt = _CLAIM_TEMPLATE.format(passages=passages)
    raw = await llm_call(prompt)

    # Robust JSON extraction
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        raw_claims = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse claim extraction response: %s", text[:200])
        return {"claims": state.get("claims", [])}

    # Grounding verification — drop hallucinated claims
    from agents.rag import get_collection

    collection = get_collection()
    verified_claims: list[Claim] = []

    for c in raw_claims:
        claim = Claim(
            claim_text=c.get("claim_text", ""),
            supporting_sources=c.get("supporting_sources", []),
            confidence=c.get("confidence", 0.5),
            conflicting_claims=[],
        )
        grounded = await verify_claim_grounding(claim, collection)
        if grounded:
            verified_claims.append(claim)
        else:
            logger.info("Dropped ungrounded claim: %.60s…", claim.claim_text)

    logger.info(
        "Extracted %d claims (%d dropped as ungrounded)",
        len(verified_claims),
        len(raw_claims) - len(verified_claims),
    )
    return {"claims": state.get("claims", []) + verified_claims}


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Conflict Detection
# ═══════════════════════════════════════════════════════════════════════════════

_CONFLICT_TEMPLATE = (_PROMPT_DIR / "conflict_detection.txt").read_text(
    encoding="utf-8"
)


async def synthesis_detect_conflicts(state: ResearchState) -> dict:
    """Identify contradictions among extracted claims."""
    claims = state.get("claims", [])
    if len(claims) < 2:
        return {
            "contradictions": [],
            "needs_more_info": False,
            "follow_up_queries": [],
            "evidence_quality": _build_evidence_quality(state),
        }

    claims_json = json.dumps(
        [
            {
                "index": i,
                "text": c.claim_text,
                "sources": c.supporting_sources,
                "confidence": c.confidence,
            }
            for i, c in enumerate(claims)
        ],
        indent=2,
    )

    prompt = _CONFLICT_TEMPLATE.format(claims_json=claims_json)
    raw = await llm_call(prompt)

    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        conflicts = json.loads(text)
    except json.JSONDecodeError:
        logger.error("Failed to parse conflict-detection response: %s", text[:200])
        conflicts = []

    # Determine if more information would help resolve inconclusive conflicts
    unresolved = [c for c in conflicts if c.get("which_is_stronger") == "inconclusive"]
    needs_more = (
        len(unresolved) > 0
        and not state.get("search_exhausted", False)
        and state.get("current_search_round", 0) < state.get("max_search_rounds", 3)
    )

    follow_up: list[str] = []
    if needs_more:
        for conflict in unresolved[:3]:
            idx_a = conflict.get("claim_a_index", 0)
            if idx_a < len(claims):
                follow_up.append(
                    f"Evidence for or against: {claims[idx_a].claim_text}"
                )

    evidence_quality = _build_evidence_quality(state)

    logger.info(
        "Found %d contradictions (%d inconclusive)", len(conflicts), len(unresolved)
    )
    return {
        "contradictions": conflicts,
        "needs_more_info": needs_more,
        "follow_up_queries": follow_up,
        "evidence_quality": evidence_quality,
    }


def _build_evidence_quality(state: ResearchState) -> dict:
    """Build a per-source evidence quality summary."""
    quality: dict[str, dict] = {}
    claims = state.get("claims", [])
    for source in state.get("search_results", []):
        supporting = [c for c in claims if source.url in c.supporting_sources]
        quality[source.url] = {
            "title": source.title,
            "credibility": source.credibility_score,
            "source_type": source.source_type,
            "claims_supported": len(supporting),
            "is_paywalled": source.is_paywalled,
        }
    return quality


# ═══════════════════════════════════════════════════════════════════════════════
#  3. Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

_REPORT_TEMPLATE = (_PROMPT_DIR / "report_generation.txt").read_text(encoding="utf-8")


async def synthesis_generate_report(state: ResearchState) -> dict:
    """Produce the final structured Markdown research report."""
    claims = state.get("claims", [])
    contradictions = state.get("contradictions", [])
    evidence_quality = state.get("evidence_quality", {})

    prompt = _REPORT_TEMPLATE.format(
        topic=state["topic"],
        scope=state.get("scope", "moderate"),
        claims_json=json.dumps([c.model_dump() for c in claims], indent=2),
        contradictions_json=json.dumps(contradictions, indent=2),
        evidence_json=json.dumps(evidence_quality, indent=2),
    )

    try:
        # Single attempt with timeout — if it fails, use fallback immediately
        # (Don't go through the full retry loop for report generation)
        import asyncio
        from utils.llm import _get_client, _rate_limit
        from config import settings as _settings

        await _rate_limit()
        model_name = _settings.primary_model
        client = _get_client()

        response = await asyncio.wait_for(
            asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=prompt,
            ),
            timeout=90.0,  # 90 second hard timeout
        )
        report_md = response.text or ""
        if not report_md.strip():
            raise ValueError("Empty response from LLM")
    except Exception as exc:
        logger.error("Failed to generate report via LLM: %s", exc)
        logger.info("Generating fallback deterministic report instead.")
        report_md = _generate_fallback_report(state)

    # Compute overall confidence
    if claims:
        avg_conf = sum(c.confidence for c in claims) / len(claims)
    else:
        avg_conf = 0.0

    if avg_conf > 0.8:
        overall = "HIGH"
    elif avg_conf > 0.5:
        overall = "MODERATE"
    else:
        overall = "LOW"

    # Collect warnings
    warnings: list[str] = []
    paywalled_count = sum(
        1
        for s in state.get("search_results", [])
        if s.is_paywalled
    )
    if paywalled_count:
        warnings.append(f"{paywalled_count} source(s) were paywalled (snippet only)")
    if state.get("search_exhausted"):
        warnings.append("Search was exhausted before full coverage was achieved")
    low_cred = [
        s for s in state.get("search_results", []) if s.credibility_score < 0.5
    ]
    if len(low_cred) > len(state.get("search_results", [])) * 0.5:
        warnings.append("Majority of sources have low credibility scores")

    report_metadata = {
        "topic": state["topic"],
        "scope": state.get("scope", "moderate"),
        "total_sources_discovered": len(state.get("search_results", [])),
        "total_sources_used": len(
            {c.source_url for c in state.get("retrieval_results", [])}
        ),
        "total_claims_extracted": len(claims),
        "total_contradictions_found": len(contradictions),
        "search_rounds_completed": state.get("current_search_round", 0),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_confidence": overall,
        "warnings": warnings,
    }

    logger.info("Report generated — %s confidence, %d warnings", overall, len(warnings))
    return {
        "report": report_md,
        "report_metadata": report_metadata,
        "synthesis_complete": True,
    }


def _generate_fallback_report(state: ResearchState) -> str:
    """Generate a deterministic report from claims when LLM fails."""
    topic = state["topic"]
    claims = state.get("claims", [])
    contradictions = state.get("contradictions", [])

    lines = [
        f"# Research Report: {topic}",
        "\n> **Note:** This report was generated automatically from extracted claims due to high load on the AI synthesis engine.\n",
        "## Key Findings",
    ]

    if not claims:
        lines.append("No verificable claims were extracted for this topic.")
    else:
        # Group claims by some simple heuristic or just list them
        for i, claim in enumerate(sorted(claims, key=lambda c: c.confidence, reverse=True), 1):
            lines.append(f"### {i}. {claim.claim_text}")
            lines.append(f"**Confidence:** {claim.confidence:.2f}")
            if claim.supporting_sources:
                lines.append("**Sources:**")
                for src in claim.supporting_sources:
                    lines.append(f"- {src}")
            lines.append("")

    if contradictions:
        lines.append("## Conflicting Evidence")
        for i, conflict in enumerate(contradictions, 1):
            lines.append(f"### Conflict {i}")
            lines.append(f"- **Claim A:** {conflict.get('claim_a', 'N/A')}")
            lines.append(f"- **Claim B:** {conflict.get('claim_b', 'N/A')}")
            lines.append(f"- **Status:** {conflict.get('which_is_stronger', 'Unknown')}")
            lines.append("")

    lines.append("## Sources Evaluated")
    for src in state.get("search_results", []):
        lines.append(f"- [{src.title}]({src.url}) (Credibility: {src.credibility_score})")

    return "\n".join(lines)
