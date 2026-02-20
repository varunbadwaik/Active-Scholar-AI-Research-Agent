"""Active Scholar — Pydantic models and LangGraph state schema."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain Models
# ═══════════════════════════════════════════════════════════════════════════════


class SourceMetadata(BaseModel):
    """A single source discovered during the search phase."""

    url: str
    title: str
    domain: str
    discovered_at: str  # ISO-8601
    credibility_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type: Literal["academic", "news", "blog", "government", "unknown"] = (
        "unknown"
    )
    is_paywalled: bool = False
    raw_snippet: str = ""


class DocumentChunk(BaseModel):
    """An embedded text chunk linked back to its source."""

    chunk_id: str
    source_url: str
    text: str
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class Claim(BaseModel):
    """A discrete, verifiable claim extracted by the synthesis agent."""

    claim_text: str
    supporting_sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    conflicting_claims: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  I/O Models (API surface)
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchRequest(BaseModel):
    """Payload the user sends to start a research investigation."""

    topic: str = Field(
        ...,
        description="The research question or topic to investigate.",
        examples=["What is the current state of nuclear fusion energy research?"],
    )
    scope: Literal["narrow", "moderate", "broad"] = Field(
        default="moderate",
        description=(
            "narrow: 3 queries, focused results. "
            "moderate: 5 queries, balanced coverage. "
            "broad: 8 queries, comprehensive sweep."
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Filters for source selection.",
        examples=[["only peer-reviewed", "published after 2023", "English only"]],
    )
    max_search_rounds: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum number of search-analyse cycles.",
    )
    mode: Literal["quick", "deep"] = Field(
        default="quick",
        description=(
            "quick: fast report (~30s, search snippets + 1 LLM call). "
            "deep: thorough analysis (~5min, full RAG pipeline)."
        ),
    )


class ReportMetadata(BaseModel):
    """Metadata attached to every generated report."""

    topic: str
    scope: str
    total_sources_discovered: int
    total_sources_used: int
    total_claims_extracted: int
    total_contradictions_found: int
    search_rounds_completed: int
    generated_at: str  # ISO-8601
    overall_confidence: Literal["HIGH", "MODERATE", "LOW"]
    warnings: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final deliverable produced by Active Scholar."""

    report_markdown: str
    metadata: ReportMetadata
    claims: list[Claim]
    contradictions: list[dict]
    sources: list[SourceMetadata]


# ═══════════════════════════════════════════════════════════════════════════════
#  LangGraph State
# ═══════════════════════════════════════════════════════════════════════════════


class ResearchState(TypedDict, total=False):
    """Shared state flowing through the LangGraph orchestration graph."""

    # ── User inputs ──
    topic: str
    scope: str  # "narrow" | "moderate" | "broad"
    constraints: list[str]
    max_search_rounds: int

    # ── Search Agent ──
    search_queries: list[str]
    search_results: list[SourceMetadata]
    explored_urls: set[str]
    current_search_round: int
    search_exhausted: bool

    # ── RAG Layer ──
    ingested_chunks: list[DocumentChunk]
    retrieval_results: list[DocumentChunk]

    # ── Synthesis Agent ──
    claims: list[Claim]
    contradictions: list[dict]
    evidence_quality: dict
    synthesis_complete: bool
    needs_more_info: bool
    follow_up_queries: list[str]

    # ── Output ──
    report: str | None
    report_metadata: dict | None
