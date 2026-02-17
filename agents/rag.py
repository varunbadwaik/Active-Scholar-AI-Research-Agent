"""Active Scholar — RAG Layer nodes.

Nodes
-----
- ``rag_ingest``   – Fetches, chunks, embeds, and stores documents.
- ``rag_retrieve`` – Retrieves the most relevant chunks for the current query.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


from config import settings
from state import DocumentChunk, ResearchState
from utils.parsing import fetch_full_text, url_hash
from utils.vectorstore import VectorStore

logger = logging.getLogger(__name__)

# ── Shared singletons (lazy-initialised) ─────────────────────────────────────

_collection: VectorStore | None = None
_embeddings: HuggingFaceEmbeddings | None = None
_splitter: RecursiveCharacterTextSplitter | None = None


def _get_collection() -> VectorStore:
    global _collection
    if _collection is None:
        _collection = VectorStore(
            persist_path=settings.chroma_persist_dir,
            collection_name="research_docs",
        )
    return _collection


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
        )
        logger.info("Local embedding model loaded: all-MiniLM-L6-v2")
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


# ── Embedding helper (local model — no rate limits) ─────────────────────────


async def _embed_with_retry(text: str) -> list[float]:
    """Embed text using local HuggingFace model. No API limits."""
    embedder = _get_embeddings()
    return await asyncio.to_thread(embedder.embed_query, text)


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
