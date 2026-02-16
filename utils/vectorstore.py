"""Active Scholar — Lightweight in-memory vector store.

Replaces ChromaDB to avoid C-extension and Python-version compatibility
issues.  Uses numpy cosine-similarity for retrieval.  Persists to a JSON
file so data survives restarts.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from threading import Lock

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Simple cosine-similarity vector store backed by numpy arrays.

    Parameters
    ----------
    persist_path:
        Directory where the store is persisted as ``vectors.json``.
        Set to ``None`` for a purely ephemeral (in-memory) store.
    collection_name:
        Logical namespace inside the store.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        collection_name: str = "default",
    ) -> None:
        self._lock = Lock()
        self._collection = collection_name
        self._persist_file: Path | None = None

        # Storage
        self._ids: list[str] = []
        self._documents: list[str] = []
        self._embeddings: list[list[float]] = []
        self._metadatas: list[dict] = []

        if persist_path:
            p = Path(persist_path)
            p.mkdir(parents=True, exist_ok=True)
            self._persist_file = p / f"{collection_name}.json"
            self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._persist_file and self._persist_file.exists():
            try:
                data = json.loads(self._persist_file.read_text(encoding="utf-8"))
                self._ids = data.get("ids", [])
                self._documents = data.get("documents", [])
                self._embeddings = data.get("embeddings", [])
                self._metadatas = data.get("metadatas", [])
                logger.info(
                    "Loaded %d vectors from %s", len(self._ids), self._persist_file
                )
            except Exception as exc:
                logger.warning("Failed to load vector store: %s", exc)

    def _save(self) -> None:
        if self._persist_file:
            data = {
                "ids": self._ids,
                "documents": self._documents,
                "embeddings": self._embeddings,
                "metadatas": self._metadatas,
            }
            self._persist_file.write_text(
                json.dumps(data), encoding="utf-8"
            )

    # ── Public API (mirrors the subset of ChromaDB we use) ───────────────────

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add vectors to the store (skip duplicates)."""
        with self._lock:
            existing = set(self._ids)
            for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
                if id_ in existing:
                    continue
                self._ids.append(id_)
                self._documents.append(doc)
                self._embeddings.append(emb)
                self._metadatas.append(meta)
                existing.add(id_)
            self._save()

    def get(self, ids: list[str]) -> dict:
        """Get documents by ID."""
        with self._lock:
            result_ids = []
            result_docs = []
            result_metas = []
            id_set = set(ids)
            for i, stored_id in enumerate(self._ids):
                if stored_id in id_set:
                    result_ids.append(stored_id)
                    result_docs.append(self._documents[i])
                    result_metas.append(self._metadatas[i])
            return {
                "ids": result_ids,
                "documents": result_docs,
                "metadatas": result_metas,
            }

    def query(
        self,
        query_embeddings: list[list[float]] | None = None,
        query_texts: list[str] | None = None,
        n_results: int = 10,
        include: list[str] | None = None,
    ) -> dict:
        """Find the most similar vectors by cosine similarity.

        Accepts either ``query_embeddings`` or ``query_texts``.  When
        ``query_texts`` is provided without embeddings a simple text-overlap
        heuristic is used (useful for grounding verification where semantic
        precision is less critical).
        """
        with self._lock:
            if not self._ids:
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            if query_embeddings:
                q = np.array(query_embeddings[0], dtype=np.float32)
                mat = np.array(self._embeddings, dtype=np.float32)

                # Cosine similarity → distance
                q_norm = q / (np.linalg.norm(q) + 1e-10)
                mat_norms = mat / (
                    np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
                )
                similarities = mat_norms @ q_norm
                distances = 1.0 - similarities  # cosine distance

                top_k = min(n_results, len(distances))
                top_indices = np.argsort(distances)[:top_k]

            elif query_texts:
                # Text-overlap fallback (for grounding checks)
                query_lower = query_texts[0].lower()
                scores = []
                for doc in self._documents:
                    # Simple word-overlap ratio
                    qwords = set(query_lower.split())
                    dwords = set(doc.lower().split())
                    overlap = len(qwords & dwords) / (len(qwords) + 1)
                    scores.append(overlap)

                scores_arr = np.array(scores)
                distances_arr = 1.0 - scores_arr
                top_k = min(n_results, len(scores_arr))
                top_indices = np.argsort(distances_arr)[:top_k]
                distances = distances_arr
            else:
                return {
                    "ids": [[]],
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                }

            r_ids = [self._ids[i] for i in top_indices]
            r_docs = [self._documents[i] for i in top_indices]
            r_metas = [self._metadatas[i] for i in top_indices]
            r_dists = [float(distances[i]) for i in top_indices]

            return {
                "ids": [r_ids],
                "documents": [r_docs],
                "metadatas": [r_metas],
                "distances": [r_dists],
            }

    @property
    def count(self) -> int:
        return len(self._ids)
