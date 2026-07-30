"""RAG orchestration tests that don't need a live LLM/retriever."""

from __future__ import annotations

import pytest

from app import rag
from app.config import get_settings
from app.providers import LLMResult
from app.rag import parse_citations


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


def test_parse_citations_accepts_paragraph_word_forms() -> None:
    text = "a [Doc A, paragraph 82] b [Doc B, paras 3, 4] c [Doc C, para. 9]"
    parsed = {(c.doc_title, c.paras) for c in parse_citations(text)}
    assert ("Doc A", "82") in parsed  # "paragraph"
    assert ("Doc B", "3, 4") in parsed  # "paras"
    assert ("Doc C", "9") in parsed  # "para." (unchanged)


def test_answer_stream_raises_if_stream_has_no_final_result(monkeypatch) -> None:
    def bad_stream(messages, *, model=None, max_tokens=None):  # noqa: ANN001, ANN202
        yield "partial token"
        return None  # generator returns no LLMResult

    monkeypatch.setattr(rag, "stream_complete", bad_stream)
    gen = rag.answer_stream("q", retriever=_EmptyRetriever(), settings=get_settings())
    with pytest.raises(RuntimeError, match="without returning an LLMResult"):
        list(gen)


# --- truncation is surfaced, not silent (T14 via T11's migration) ----------- #
def test_answer_reports_truncation_from_the_provider(monkeypatch) -> None:
    """Hitting MAX_ANSWER_TOKENS leaves the text cut off mid-sentence. Without a
    flag the answer looks complete and is quietly wrong."""
    import app.rag as rag
    from app.providers import LLMResult
    from app.rewrite import RewriteResult

    monkeypatch.setattr(
        rag, "_prepare", lambda *a, **k: (RewriteResult("q", "q", False), [], [])
    )
    for reason, expected in (("length", True), ("stop", False)):
        monkeypatch.setattr(
            rag, "complete",
            lambda *a, _r=reason, **k: LLMResult(
                "partial answer", "gpt-4o-mini", 10, 5, 0.0, 1, finish_reason=_r
            ),
        )
        assert rag.answer("q").truncated is expected
