"""Active Scholar — LangGraph state-machine orchestration.

Defines the research workflow as a compiled LangGraph ``StateGraph``
with conditional looping between the search and synthesis phases.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.rag import rag_ingest, rag_retrieve
from agents.search import (
    search_evaluate_credibility,
    search_execute,
    search_generate_queries,
)
from agents.synthesis import (
    synthesis_analyze,
    synthesis_detect_conflicts,
    synthesis_generate_report,
)
from state import ResearchState

# ═══════════════════════════════════════════════════════════════════════════════
#  Routing
# ═══════════════════════════════════════════════════════════════════════════════


def _route_after_synthesis(state: ResearchState) -> str:
    """Decide whether to loop back for more searching or produce the report."""
    if (
        state.get("needs_more_info", False)
        and state.get("current_search_round", 0) < state.get("max_search_rounds", 3)
        and not state.get("search_exhausted", False)
    ):
        return "need_more_search"
    return "generate_report"


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph Construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_graph() -> StateGraph:
    """Construct and compile the Active Scholar research graph."""
    builder = StateGraph(ResearchState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    builder.add_node("generate_queries", search_generate_queries)
    builder.add_node("execute_search", search_execute)
    builder.add_node("evaluate_sources", search_evaluate_credibility)
    builder.add_node("ingest_documents", rag_ingest)
    builder.add_node("retrieve_context", rag_retrieve)
    builder.add_node("analyze_claims", synthesis_analyze)
    builder.add_node("detect_conflicts", synthesis_detect_conflicts)
    builder.add_node("generate_report", synthesis_generate_report)

    # ── Edges ────────────────────────────────────────────────────────────────
    builder.set_entry_point("generate_queries")

    builder.add_edge("generate_queries", "execute_search")
    builder.add_edge("execute_search", "evaluate_sources")
    builder.add_edge("evaluate_sources", "ingest_documents")
    builder.add_edge("ingest_documents", "retrieve_context")
    builder.add_edge("retrieve_context", "analyze_claims")
    builder.add_edge("analyze_claims", "detect_conflicts")

    # ── Conditional loop ─────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "detect_conflicts",
        _route_after_synthesis,
        {
            "need_more_search": "generate_queries",
            "generate_report": "generate_report",
        },
    )
    builder.add_edge("generate_report", END)

    return builder.compile()


# Pre-built graph singleton
graph = build_graph()
