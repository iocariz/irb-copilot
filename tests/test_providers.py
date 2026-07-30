"""Provider tests: cross-family parameter handling, pricing, cost accounting.

The GPT-4o and GPT-5 families disagree on two request parameters, and getting
either wrong is a hard 400 at runtime (verified against the live API):

* every GPT-5 model rejects the legacy ``max_tokens`` and needs
  ``max_completion_tokens`` (which GPT-4o also accepts, so one path serves both);
* some GPT-5 models reject any ``temperature`` other than their default.

These tests pin that behaviour so a model swap can't silently regress it.
"""

from __future__ import annotations

import pytest

from app.providers import (
    FIXED_TEMPERATURE_MODELS,
    PRICING,
    LLMResult,
    _reasoning_tokens,
    _request_kwargs,
    estimate_cost,
    require_pricing,
    unpriced_models,
)


# --- request parameters ----------------------------------------------------- #
def test_uses_max_completion_tokens_not_legacy_max_tokens() -> None:
    kwargs = _request_kwargs("gpt-5.4-mini", 0.0, 1500)
    assert kwargs["max_completion_tokens"] == 1500
    assert "max_tokens" not in kwargs  # the legacy name 400s on every GPT-5 model


def test_max_tokens_omitted_entirely_when_unset() -> None:
    assert "max_completion_tokens" not in _request_kwargs("gpt-4o-mini", 0.0, None)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "gpt-5.4-mini", "gpt-5.4"])
def test_temperature_sent_for_models_that_accept_it(model: str) -> None:
    assert _request_kwargs(model, 0.0, 100)["temperature"] == 0.0


@pytest.mark.parametrize("model", sorted(FIXED_TEMPERATURE_MODELS))
def test_temperature_omitted_for_fixed_temperature_models(model: str) -> None:
    # Sending temperature=0.0 to these returns 400; omitting it is the only
    # way to call them at all.
    assert "temperature" not in _request_kwargs(model, 0.0, 100)


# --- pricing ---------------------------------------------------------------- #
def test_estimate_cost_uses_the_pricing_table() -> None:
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out.
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)


def test_unknown_model_costs_zero_but_is_reported_as_unpriced() -> None:
    assert estimate_cost("some-future-model", 1_000_000, 1_000_000) == 0.0
    assert unpriced_models(["some-future-model"]) == ["some-future-model"]


def test_require_pricing_raises_for_unpriced_models() -> None:
    # A missing price silently zeroes the eval's cost columns, so callers that
    # publish cost figures must fail fast instead.
    with pytest.raises(RuntimeError, match="no pricing for some-future-model"):
        require_pricing(["gpt-4o-mini", "some-future-model"])


def test_require_pricing_passes_for_configured_defaults() -> None:
    from app.config import get_settings

    settings = get_settings()
    require_pricing([settings.llm_model, settings.judge_model, settings.embedding_model])
    require_pricing(settings.eval_models_list)


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "gpt-5.4-mini", "gpt-5-mini"])
def test_pricing_entries_are_positive_pairs(model: str) -> None:
    price_in, price_out = PRICING[model]
    assert price_in > 0 and price_out > 0


# --- reasoning tokens ------------------------------------------------------- #
class _Details:
    def __init__(self, reasoning_tokens: int) -> None:
        self.reasoning_tokens = reasoning_tokens


class _Usage:
    def __init__(self, details: object | None) -> None:
        self.completion_tokens_details = details


def test_reasoning_tokens_read_from_usage() -> None:
    assert _reasoning_tokens(_Usage(_Details(192))) == 192


@pytest.mark.parametrize("usage", [_Usage(None), _Usage(_Details(0)), object()])
def test_reasoning_tokens_default_to_zero(usage: object) -> None:
    # Non-reasoning models omit the field entirely; that must read as 0, not fail.
    assert _reasoning_tokens(usage) == 0


def test_reasoning_tokens_are_not_double_counted_in_cost() -> None:
    # Reasoning tokens are already part of completion_tokens and priced at the
    # output rate, so cost depends only on tokens_out.
    result = LLMResult(
        text="x", model="gpt-5.4-mini", tokens_in=100, tokens_out=500,
        cost_usd=estimate_cost("gpt-5.4-mini", 100, 500), latency_ms=1,
        reasoning_tokens=400,
    )
    assert result.cost_usd == pytest.approx(estimate_cost("gpt-5.4-mini", 100, 500))
    assert result.reasoning_tokens < result.tokens_out


# --- finish_reason / truncation ---------------------------------------------- #
def test_truncated_is_true_only_for_a_length_stop() -> None:
    def result(reason: str) -> LLMResult:
        return LLMResult("t", "gpt-4o-mini", 1, 1, 0.0, 1, finish_reason=reason)

    assert result("length").truncated is True
    assert result("stop").truncated is False
    assert result("").truncated is False  # unknown/absent must not read as truncated


def test_finish_reason_defaults_to_empty() -> None:
    # Older call sites construct LLMResult without it; that must not break.
    assert LLMResult("t", "gpt-4o-mini", 1, 1, 0.0, 1).finish_reason == ""


# --- query-embedding cache (T7) ---------------------------------------------- #
@pytest.fixture(autouse=False)
def _clear_embed_cache():
    from app.providers import _embed_cache

    _embed_cache.clear()
    yield
    _embed_cache.clear()


def test_embed_query_calls_the_api_once_per_distinct_string(monkeypatch, _clear_embed_cache):
    import app.providers as providers

    calls: list[list[str]] = []
    monkeypatch.setattr(
        providers, "embed_texts", lambda texts: calls.append(texts) or [[0.1, 0.2]] * len(texts)
    )
    for _ in range(5):
        assert providers.embed_query("what is LGD?") == [0.1, 0.2]
    assert len(calls) == 1  # 5 searches, 1 API call


def test_embed_query_returns_a_copy_so_callers_cannot_poison_the_cache(
    monkeypatch, _clear_embed_cache
):
    import app.providers as providers

    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[0.1, 0.2]] * len(texts))
    first = providers.embed_query("q")
    first.append(999.0)  # a caller mutating its result must not corrupt the cache
    assert providers.embed_query("q") == [0.1, 0.2]


def test_warm_query_embeddings_batches_and_dedupes(monkeypatch, _clear_embed_cache):
    import app.providers as providers

    calls: list[list[str]] = []
    monkeypatch.setattr(
        providers, "embed_texts", lambda texts: calls.append(texts) or [[0.5]] * len(texts)
    )
    warmed = providers.warm_query_embeddings(["a", "b", "a", "c"])
    assert warmed == 3  # deduped
    assert len(calls) == 1 and calls[0] == ["a", "b", "c"]  # one batched request
    # Subsequent searches are served from the cache.
    calls.clear()
    assert providers.embed_query("b") == [0.5]
    assert calls == []


def test_warm_query_embeddings_skips_already_cached(monkeypatch, _clear_embed_cache):
    import app.providers as providers

    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[0.5]] * len(texts))
    providers.warm_query_embeddings(["a"])
    assert providers.warm_query_embeddings(["a"]) == 0


def test_embed_cache_is_bounded(monkeypatch, _clear_embed_cache):
    import app.providers as providers

    monkeypatch.setattr(providers, "embed_texts", lambda texts: [[0.5]] * len(texts))
    monkeypatch.setattr(providers, "_EMBED_CACHE_MAX", 3)
    for i in range(10):
        providers.embed_query(f"q{i}")
    assert len(providers._embed_cache) <= 3  # LRU eviction keeps memory bounded


def test_cache_is_keyed_by_embedding_model(monkeypatch, _clear_embed_cache):
    import app.providers as providers

    calls: list[list[str]] = []
    monkeypatch.setattr(
        providers, "embed_texts", lambda texts: calls.append(texts) or [[0.1]] * len(texts)
    )
    providers.embed_query("q")
    monkeypatch.setattr(providers.settings, "embedding_model", "text-embedding-3-large")
    providers.embed_query("q")  # different model -> must not reuse the vector
    assert len(calls) == 2


# --- quota exhaustion is permanent, not transient ---------------------------- #
def test_quota_exhaustion_is_distinguished_from_a_rate_limit() -> None:
    """Both arrive as HTTP 429, but one is transient and one is terminal.
    Treating billing exhaustion as retryable yields a biased subsample."""
    from app.providers import is_quota_exhausted

    quota = Exception(
        "Error code: 429 - You exceeded your current quota, please check your plan"
    )
    tpm = Exception(
        "Error code: 429 - Rate limit reached for gpt-4o on tokens per min (TPM): "
        "Limit 30000, Used 25635"
    )
    assert is_quota_exhausted(quota) is True
    assert is_quota_exhausted(tpm) is False


def test_raise_if_quota_exhausted_converts_only_quota_errors() -> None:
    from app.providers import QuotaExhausted, raise_if_quota_exhausted

    with pytest.raises(QuotaExhausted, match="top up billing"):
        raise_if_quota_exhausted(Exception("You exceeded your current quota"))
    # A transient rate limit must pass through untouched, so the caller's own
    # per-item handling still applies.
    raise_if_quota_exhausted(Exception("Rate limit reached on tokens per min (TPM)"))
