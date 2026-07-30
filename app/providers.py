"""Model provider factory (SPEC §2, §17).

All model access goes through this module so providers can be swapped via env
vars, and every LLM call is returned with token counts and cost so callers can
log them (SPEC §17). Pricing is a small, editable table; unknown models cost 0
with a warning rather than failing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_EMBED_BATCH = 128
# Query-embedding cache. Bounded so a long-running API process can't grow it
# without limit: 2048 x 1536 float64 is ~25 MB, and it comfortably holds the
# retrieval evaluation's full working set (800 questions x 3 rewrite modes).
_EMBED_CACHE_MAX = 2048
_embed_cache: OrderedDict[tuple[str, str], tuple[float, ...]] = OrderedDict()
# lru_cache would be simpler but cannot be pre-populated in batches, which is
# where most of the saving is. Guard the OrderedDict: retrieval runs under
# FastAPI's thread pool and the evaluation's worker pools.
_embed_lock = threading.Lock()
# The SDK auto-retries 429/5xx with exponential backoff honouring Retry-After;
# the default of 2 is too few for sustained token-per-minute limits under the
# eval's concurrent fan-out, so allow more attempts to ride out the window.
_MAX_RETRIES = 8

# USD per 1M tokens (input, output). Embeddings use the input column only.
# Approximate list prices — adjust as needed; keep models here to keep cost math
# out of business logic. A model missing from this table records cost 0, which
# would silently corrupt the eval's cost columns and the Grafana cost panel —
# see `require_pricing`, which fails fast instead.
PRICING: dict[str, tuple[float, float]] = {
    # GPT-4o family
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # GPT-5 family
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-5-pro": (15.00, 120.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol": (5.00, 30.00),
    # Embeddings
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

# Models that reject any `temperature` other than their default — verified
# against the API: gpt-5-mini returns 400 "'temperature' does not support 0.0".
# For these we omit the parameter entirely rather than fail the call. Note this
# costs determinism, so they are a poor fit for the evaluation (which relies on
# temperature=0 for reproducibility); gpt-5.4-* and gpt-4o-* do accept 0.0.
FIXED_TEMPERATURE_MODELS: frozenset[str] = frozenset(
    {"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-pro"}
)

# Models whose token-per-minute limit is low enough that a batch caller must pace
# itself, mapped to the maximum requests it may have in flight.
#
# OpenAI meters TPM using `max_tokens` as the output estimate, not actual usage,
# so one answer request reserves ~2.1k prompt + 1.5k cap ≈ 3.6k tokens. gpt-4o at
# 30k TPM therefore sustains only ~8 requests/minute. Running 3 concurrent
# workers attempted ~45/min: every retry's freed budget was immediately consumed
# by a sibling worker, so the SDK's backoff never converged and requests failed
# after all 8 attempts (a thundering herd, not a transient blip).
#
# Serialising the offending model lets the backoff actually work. Models absent
# from this map are unconstrained.
MODEL_MAX_CONCURRENCY: dict[str, int] = {"gpt-4o": 1}


def max_concurrency(model: str, requested: int) -> int:
    """Requests-in-flight budget for `model`, never above `requested`."""
    return max(1, min(requested, MODEL_MAX_CONCURRENCY.get(model, requested)))


class QuotaExhausted(RuntimeError):
    """The account is out of credit — every further request will also fail."""


# Billing exhaustion arrives as HTTP 429, the same status as a token-per-minute
# rate limit, but it is permanent rather than transient: retrying and skipping
# ahead only burns time and silently shrinks the sample. The response carries no
# machine-readable `code` for this case (verified against the live API), so match
# the message.
_QUOTA_MARKERS = ("exceeded your current quota", "insufficient_quota")


def is_quota_exhausted(exc: BaseException) -> bool:
    """True if `exc` is billing exhaustion rather than a transient rate limit."""
    return any(marker in str(exc).lower() for marker in _QUOTA_MARKERS)


def raise_if_quota_exhausted(exc: BaseException) -> None:
    """Convert a billing-exhaustion error into `QuotaExhausted`.

    Callers that tolerate per-item failures must still abort on this: a
    partially-completed configuration is not a smaller valid sample, it is a
    biased one (whichever items happened to run before the credit ran out).
    """
    if is_quota_exhausted(exc):
        raise QuotaExhausted(
            "OpenAI account quota exhausted — top up billing and re-run. "
            "Aborting so no partial, non-random sample is written."
        ) from exc

# Warn at most once per unknown model instead of on every call.
_WARNED_UNPRICED: set[str] = set()


@dataclass(frozen=True)
class LLMResult:
    """The outcome of one chat completion, with accounting.

    ``reasoning_tokens`` is the share of ``tokens_out`` a reasoning model spent
    thinking rather than answering (0 for non-reasoning models). It is already
    *included* in ``tokens_out`` and priced at the output rate, so it is recorded
    for visibility only — never added to the cost separately.

    ``finish_reason`` is the provider's reason for stopping. ``"length"`` means
    the output hit the token cap and the text is cut off mid-stream — which for
    structured output (the LLM judge) is indistinguishable from a bad answer
    unless callers can see it. Use ``truncated``.
    """

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    reasoning_tokens: int = 0
    finish_reason: str = ""

    @property
    def truncated(self) -> bool:
        """True if generation stopped because it ran out of token budget."""
        return self.finish_reason == "length"


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
        if model not in _WARNED_UNPRICED:  # warn once, not per call
            _WARNED_UNPRICED.add(model)
            logger.warning(
                "no pricing for model %r; cost recorded as 0 — add it to "
                "app.providers.PRICING so cost tracking stays meaningful",
                model,
            )
        return 0.0
    in_price, out_price = PRICING[model]
    return (tokens_in * in_price + tokens_out * out_price) / 1_000_000


def unpriced_models(models: Iterable[str]) -> list[str]:
    """Which of `models` have no entry in `PRICING` (pure, sorted)."""
    return sorted({m for m in models if m not in PRICING})


def require_pricing(models: Iterable[str]) -> None:
    """Fail fast if any model lacks pricing.

    A missing entry makes `estimate_cost` return 0, which does not raise but
    silently zeroes the cost columns in the evaluation CSVs and the Grafana cost
    panel — a corrupt result is worse than a crash, so callers that publish cost
    figures (the evaluations) call this up front.
    """
    missing = unpriced_models(models)
    if missing:
        raise RuntimeError(
            f"no pricing for {', '.join(missing)} — add entries to "
            f"app.providers.PRICING before running, or cost will be recorded as $0"
        )


def _request_kwargs(model: str, temperature: float, max_tokens: int | None) -> dict:
    """Build the provider-specific chat-completion parameters.

    Two cross-family quirks are handled here so callers stay uniform:

    * ``max_completion_tokens`` — every GPT-5 model rejects the legacy
      ``max_tokens`` ("Unsupported parameter"); the GPT-4o family accepts the
      new name too, so one code path serves both.
    * ``temperature`` — omitted for models that only allow their default (see
      ``FIXED_TEMPERATURE_MODELS``), which would otherwise 400 on our 0.0.
    """
    kwargs: dict = {}
    if model not in FIXED_TEMPERATURE_MODELS:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return kwargs


def _reasoning_tokens(usage: object) -> int:
    """Reasoning tokens from a usage object (0 when the model reports none)."""
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", 0) or 0


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
        **_request_kwargs(model, temperature, max_tokens),
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
        reasoning_tokens=_reasoning_tokens(usage),
        finish_reason=response.choices[0].finish_reason or "",
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
        stream=True,
        stream_options={"include_usage": True},
        **_request_kwargs(model, temperature, max_tokens),
    )
    parts: list[str] = []
    tokens_in = tokens_out = reasoning_tokens = 0
    finish_reason = ""
    for chunk in stream:
        if chunk.usage:  # final usage-only chunk
            tokens_in = chunk.usage.prompt_tokens
            tokens_out = chunk.usage.completion_tokens
            reasoning_tokens = _reasoning_tokens(chunk.usage)
        if chunk.choices:
            # Arrives on the last content chunk, before the usage-only one.
            finish_reason = chunk.choices[0].finish_reason or finish_reason
            if delta := chunk.choices[0].delta.content:
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
        reasoning_tokens=reasoning_tokens,
        finish_reason=finish_reason,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with the configured embedding model (batched).

    Uncached by design: the caller is either ingestion (every chunk embedded
    exactly once — caching would only waste memory) or `warm_query_embeddings`,
    which populates the query cache itself.
    """
    client = openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string, reusing a cached vector when possible.

    Retrieval embeds the query on every vector/hybrid search, so the same string
    is embedded once per configuration: the retrieval evaluation issues ~14,400
    single-item requests for only ~2,400 distinct queries. Caching removes that
    redundancy; `warm_query_embeddings` additionally collapses the remaining
    requests into batches.
    """
    key = (settings.embedding_model, text)
    if (cached := _cache_get(key)) is not None:
        return list(cached)  # copy: callers must not mutate the cached vector
    vector = embed_texts([text])[0]
    _cache_put(key, tuple(vector))
    return vector


def warm_query_embeddings(texts: Iterable[str]) -> int:
    """Pre-embed `texts` in batches and populate the query cache.

    Turns N single-item HTTP round-trips into ceil(N/128) batched ones, on top of
    the deduplication `embed_query` already does. Returns how many were newly
    embedded. Safe to call repeatedly — already-cached texts are skipped.
    """
    model = settings.embedding_model
    pending = list(dict.fromkeys(t for t in texts if _cache_get((model, t)) is None))
    if not pending:
        return 0
    for vector, text in zip(embed_texts(pending), pending, strict=True):
        _cache_put((model, text), tuple(vector))
    return len(pending)


def _cache_get(key: tuple[str, str]) -> tuple[float, ...] | None:
    with _embed_lock:
        vector = _embed_cache.get(key)
        if vector is not None:
            _embed_cache.move_to_end(key)  # LRU: mark as recently used
        return vector


def _cache_put(key: tuple[str, str], vector: tuple[float, ...]) -> None:
    with _embed_lock:
        _embed_cache[key] = vector
        _embed_cache.move_to_end(key)
        while len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)  # evict least recently used
