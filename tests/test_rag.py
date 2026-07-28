"""RAG orchestration tests that don't need a live LLM/retriever."""

from __future__ import annotations

from app import rag
from app.config import get_settings
from app.providers import LLMResult


class _EmptyRetriever:
    """Stand-in retriever that returns no chunks (enough to drive answer())."""

    def search(self, *_a, **_k):  # noqa: ANN002, ANN003, ANN201
        return []


def test_answer_caps_generation_tokens(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_complete(messages, *, model=None, max_tokens=None, **_kw):  # noqa: ANN001, ANN202
        captured["max_tokens"] = max_tokens
        return LLMResult(
            text="ok", model=model or "m", tokens_in=1, tokens_out=1,
            cost_usd=0.0, latency_ms=1,
        )

    monkeypatch.setattr(rag, "complete", fake_complete)
    settings = get_settings()
    rag.answer("q", retriever=_EmptyRetriever(), settings=settings)
    # The answer call must be bounded by the configured cap (not None/unbounded).
    assert captured["max_tokens"] == settings.max_answer_tokens
    assert captured["max_tokens"] is not None
