"""Active Scholar — Configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ActiveScholarConfig(BaseSettings):
    """All tuneable knobs for the Active Scholar pipeline.

    Values are read from environment variables prefixed with ``ACTIVE_SCHOLAR_``
    or from an ``.env`` file at the project root.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    google_api_key: str = ""
    primary_model: str = "gemini-1.5-flash"
    fallback_model: str = "gemini-1.5-pro"
    llm_temperature: float = 0.1

    # ── Search ───────────────────────────────────────────────────────────────
    tavily_api_key: str = ""
    serper_api_key: str | None = None
    max_search_rounds: int = 3
    max_results_per_query: int = 10

    # ── RAG / Embeddings ─────────────────────────────────────────────────────
    embedding_model: str = "models/text-embedding-004"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chroma_persist_dir: str = "./active_scholar_db"
    retrieval_top_k: int = 20
    relevance_threshold: float = 0.45

    # ── Credibility ──────────────────────────────────────────────────────────
    min_credibility_score: float = 0.3

    # ── Report ───────────────────────────────────────────────────────────────
    default_scope: str = "moderate"

    model_config = {
        "env_prefix": "ACTIVE_SCHOLAR_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton – import this where needed
settings = ActiveScholarConfig()
