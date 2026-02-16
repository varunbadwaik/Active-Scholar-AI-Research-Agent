"""Tests for the LangGraph orchestration graph."""

from __future__ import annotations

from graph import build_graph
from state import ResearchState


def test_graph_compiles():
    """The graph should compile without errors."""
    g = build_graph()
    assert g is not None


def test_graph_has_expected_nodes():
    """All 8 expected nodes should be present."""
    g = build_graph()
    node_names = set(g.nodes.keys())
    expected = {
        "generate_queries",
        "execute_search",
        "evaluate_sources",
        "ingest_documents",
        "retrieve_context",
        "analyze_claims",
        "detect_conflicts",
        "generate_report",
        "__start__",  # LangGraph adds this automatically
    }
    # All expected nodes should be present (graph may add __end__ too)
    assert expected.issubset(node_names | {"__start__"})
