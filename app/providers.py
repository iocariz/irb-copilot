"""Model provider factory (SPEC §2, §17).

All model access goes through this module so providers can be swapped via env
vars, and every LLM call is returned with token counts and cost so callers can
log them (SPEC §17). Pricing is a small, editable table; unknown models cost 0
with a warning rather than failing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_EMBED_BATCH = 128
# The SDK auto-retries 429/5xx with exponential backoff honouring Retry-After;
# the default of 2 is too few for sustained token-per-minute limits under the
# eval's concurrent fan-out, so allow more attempts to ride out the window.
_MAX_RETRIES = 8

# USD per 1M tokens (input, output). Embeddings use the input column only.
# Approximate list prices — adjust as needed; keep models here to keep cost math
# out of business logic.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass(frozen=True)
class LLMResult:
    """The outcome of one chat completion, with accounting."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    """Lazily construct a single OpenAI client, honouring an optional base URL."""
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — required for embeddings / LLM calls."
        )
    kwargs: dict[str, str] = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    return OpenAI(max_retries=_MAX_RETRIES, **kwargs)


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """USD cost for a call, from the pricing table (0 for unknown models)."""
    if model not in PRICING:
        logger.warning("no pricing for model %r; cost recorded as 0", model)
        return 0.0
    in_price, out_price = PRICING[model]
    return (tokens_in * in_price + tokens_out * out_price) / 1_000_000


def complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMResult:
    """Run a chat completion and return the text with token/cost/latency data."""
    model = model or settings.llm_model
    client = openai_client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = response.usage
    tokens_in = usage.prompt_tokens if usage else 0
    tokens_out = usage.completion_tokens if usage else 0
    return LLMResult(
        text=response.choices[0].message.content or "",
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=estimate_cost(model, tokens_in, tokens_out),
        latency_ms=latency_ms,
    )


def stream_complete(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> Generator[str, None, LLMResult]:
    """Stream a chat completion token-by-token.

    Yields text deltas as they arrive; the generator's return value (available
    via StopIteration.value) is the final `LLMResult` with token/cost/latency.
    The ``Generator[..., LLMResult]`` annotation makes that final value part of
    the type contract — a caller that ``for``-loops over the deltas discards it.
    """
    model = model or settings.llm_model
    client = openai_client()
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    parts: list[str] = []
    tokens_in = tokens_out = 0
    for chunk in stream:
        if chunk.usage:  # final usage-only chunk
            tokens_in = chunk.usage.prompt_tokens
            tokens_out = chunk.usage.completion_tokens
        if chunk.choices and (delta := chunk.choices[0].delta.content):
            parts.append(delta)
            yield delta
    latency_ms = int((time.perf_counter() - start) * 1000)
    return LLMResult(
        text="".join(parts),
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=estimate_cost(model, tokens_in, tokens_out),
        latency_ms=latency_ms,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with the configured embedding model (batched)."""
    client = openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
