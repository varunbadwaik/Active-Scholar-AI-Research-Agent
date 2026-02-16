"""Active Scholar — RAG Layer nodes.

Nodes
-----
- ``rag_ingest``   – Fetches, chunks, embeds, and stores documents.
- ``rag_retrieve`` – Retrieves the most relevant chunks for the current query.
"""

from __future__ import annotations

import asyncio
import logging
import time

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from state import DocumentChunk, ResearchState
from utils.parsing import fetch_full_text, url_hash
from utils.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Shared singletons (lazy-initialised) ─────────────────────────────────────

_collection: VectorStore | None = None
_embeddings: GoogleGenerativeAIEmbeddings | None = None
_splitter: RecursiveCharacterTextSplitter | None = None

# ── Embedding rate limiter ───────────────────────────────────────────────────
_embed_last_call_ts: float = 0.0
_EMBED_INTERVAL = 2.0  # seconds between embedding API calls


def _get_collection() -> VectorStore:
    global _collection
    if _collection is None:
        _collection = VectorStore(
            persist_path=settings.chroma_persist_dir,
            collection_name="research_docs",
        )
    return _collection


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
    return _embeddings


def _get_splitter() -> RecursiveCharacterTextSplitter:
    global _splitter
    if _splitter is None:
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )
    return _splitter


def get_collection() -> VectorStore:
    """Public accessor so other modules (verification) can reach the store."""
    return _get_collection()


# ── Retry wrapper for embeddings ─────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _embed_with_retry(text: str) -> list[float]:
    """Embed text with rate-limiting and retry logic for 429 errors."""
    global _embed_last_call_ts
    now = time.monotonic()
    elapsed = now - _embed_last_call_ts
    if elapsed < _EMBED_INTERVAL:
        await asyncio.sleep(_EMBED_INTERVAL - elapsed)
    _embed_last_call_ts = time.monotonic()

    embedder = _get_embeddings()
    return await embedder.aembed_query(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Ingestion
# ═══════════════════════════════════════════════════════════════════════════════


async def rag_ingest(state: ResearchState) -> dict:
    """Fetch full text for each source, chunk it, embed, and store."""
    collection = _get_collection()
    splitter = _get_splitter()

    new_chunks: list[DocumentChunk] = []

    for source in state.get("search_results", []):
        # Decide text to use
        if source.is_paywalled:
            text = source.raw_snippet
        else:
            text = await fetch_full_text(source.url)
            if not text:
                text = source.raw_snippet  # fallback to snippet

        if not text:
            logger.info("No text available for %s — skipping", source.url)
            continue

        chunks = splitter.split_text(text)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{url_hash(source.url)}_{idx}"

            # Skip duplicates already in the store
            existing = collection.get(ids=[chunk_id])
            if existing and existing["ids"]:
                continue

            try:
                embedding = await _embed_with_retry(chunk_text)
            except Exception as exc:
                logger.warning("Embedding failed for chunk %s: %s", chunk_id, exc)
                continue

            metadata = {
                "source_url": source.url,
                "source_title": source.title,
                "source_type": source.source_type,
                "credibility_score": source.credibility_score,
                "chunk_index": idx,
                "domain": source.domain,
                "discovered_at": source.discovered_at,
            }

            collection.add(
                ids=[chunk_id],
                documents=[chunk_text],
                embeddings=[embedding],
                metadatas=[metadata],
            )

            new_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_url=source.url,
                    text=chunk_text,
                    embedding=embedding,
                    metadata=metadata,
                )
            )

    logger.info("Ingested %d new chunks", len(new_chunks))
    return {
        "ingested_chunks": state.get("ingested_chunks", []) + new_chunks,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


async def rag_retrieve(state: ResearchState) -> dict:
    """Retrieve the most relevant chunks for the current research topic."""
    collection = _get_collection()

    # Build a composite retrieval query
    parts = [state["topic"]]
    follow_ups = state.get("follow_up_queries", [])
    if follow_ups:
        parts.extend(follow_ups[-3:])
    retrieval_query = " ".join(parts)

    try:
        query_embedding = await _embed_with_retry(retrieval_query)
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        return {"retrieval_results": []}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.retrieval_top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[DocumentChunk] = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        # Relevance gate
        if distance > settings.relevance_threshold:
            continue

        retrieved.append(
            DocumentChunk(
                chunk_id=results["ids"][0][i],
                source_url=meta.get("source_url", ""),
                text=doc,
                metadata={**meta, "retrieval_distance": distance},
            )
        )

    # Sort: highest credibility first, then closest distance
    retrieved.sort(
        key=lambda c: (
            -c.metadata.get("credibility_score", 0),
            c.metadata.get("retrieval_distance", 1.0),
        )
    )

    logger.info("Retrieved %d relevant chunks", len(retrieved))
    return {"retrieval_results": retrieved}
