"""Active Scholar — LLM helpers using Google Gemini (google-genai SDK).

Includes aggressive retry logic to handle Gemini free-tier rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import time

from google import genai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

# ── Client singleton ────────────────────────────────────────────────────────

_client: genai.Client | None = None
_last_call_ts: float = 0.0          # simple rate-limiter
MIN_CALL_INTERVAL = 6.0             # seconds between calls (free tier ≈ 10 RPM)


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


async def _rate_limit() -> None:
    """Wait if we're calling too fast for the free-tier quota."""
    global _last_call_ts
    now = time.monotonic()
    elapsed = now - _last_call_ts
    if elapsed < MIN_CALL_INTERVAL:
        wait = MIN_CALL_INTERVAL - elapsed
        logger.debug("Rate-limiter: waiting %.1f s", wait)
        await asyncio.sleep(wait)
    _last_call_ts = time.monotonic()


# ── Public interface ─────────────────────────────────────────────────────────


@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=10, min=10, max=300),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def llm_call(prompt: str, *, use_fallback: bool = False) -> str:
    """Invoke the LLM with automatic primary→fallback escalation.

    Uses the google-genai SDK directly for maximum compatibility with
    the latest Gemini models.  Includes a simple rate-limiter and long
    exponential back-off to handle free-tier quotas gracefully.

    Parameters
    ----------
    prompt:
        The full prompt string to send.
    use_fallback:
        If ``True``, skip the primary model and use the fallback.

    Returns
    -------
    str
        The model's text response.
    """
    model_name = settings.fallback_model if use_fallback else settings.primary_model
    client = _get_client()

    await _rate_limit()

    try:
        # google-genai's generate_content is synchronous; run in executor
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=model_name,
            contents=prompt,
        )
        return response.text or ""
    except Exception as exc:
        exc_str = str(exc)
        # On rate-limit, let tenacity retry with back-off
        if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
            logger.warning("Rate-limited on %s — will retry after back-off", model_name)
            raise
        # On other errors, try fallback model before raising
        if not use_fallback:
            logger.warning(
                "Primary LLM (%s) failed: %s. Falling back to %s",
                settings.primary_model,
                exc,
                settings.fallback_model,
            )
            return await llm_call(prompt, use_fallback=True)
        raise
