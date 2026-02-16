"""Active Scholar — Hallucination grounding verification."""

from __future__ import annotations

import logging

from state import Claim
from utils.vectorstore import VectorStore

logger = logging.getLogger(__name__)


async def verify_claim_grounding(
    claim: Claim,
    collection: VectorStore,
) -> bool:
    """Check that *claim* is grounded in at least one of its cited sources.

    Queries the vector store with the claim text and verifies that one of the
    returned chunks originates from a URL listed in ``claim.supporting_sources``.
    Returns ``False`` (and logs a warning) if the claim appears to be a
    hallucination.
    """
    try:
        results = collection.query(
            query_texts=[claim.claim_text],
            n_results=5,
            include=["metadatas"],
        )
    except Exception as exc:
        logger.error("Grounding verification query failed: %s", exc)
        return False  # fail-closed: treat unverifiable claims as ungrounded

    if not results["metadatas"] or not results["metadatas"][0]:
        logger.warning("No vector-DB matches for claim: %.80s…", claim.claim_text)
        return False

    result_urls = {m.get("source_url", "") for m in results["metadatas"][0]}
    grounded = bool(set(claim.supporting_sources) & result_urls)
    if not grounded:
        logger.warning(
            "Claim appears ungrounded (hallucination?): %.80s…", claim.claim_text
        )
    return grounded
