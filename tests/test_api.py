"""API tests (SPEC §9) using FastAPI's TestClient with the RAG flow and DB
mocked out, so no LLM, Qdrant, or Postgres is required."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api as api
from app.rag import Answer, Citation, SourceChunk


def _fake_answer() -> Answer:
    return Answer(
        text="Institutions must apply a margin of conservatism [Doc, para. 82].",
        citations=[Citation(text="Doc, para. 82", doc_title="Doc", paras="82")],
        chunks_used=[
            SourceChunk(
                chunk_id="c1", doc_id="d1", doc_title="Doc", section_path=["5"],
                para_ids=["82"], pages=[45], score=1.23, text="chunk text",
            )
        ],
        question="q", rewritten_query="q", retrieval_mode="bm25", prompt_version="v2",
        model="gpt-4o-mini", tokens_in=100, tokens_out=20, cost_usd=0.0001,
        latency_ms=500, used_rewrite=False,
    )


@pytest.fixture(autouse=True)
def _safe_defaults(monkeypatch):
    """Never touch a real DB / Qdrant during API tests."""
    monkeypatch.setattr(api, "init_db", lambda: None)
    monkeypatch.setattr(api, "ping_qdrant", lambda: True)
    monkeypatch.setattr(api, "ping_postgres", lambda: True)


@pytest.fixture
def client() -> TestClient:
    return TestClient(api.app)


def test_ask_returns_answer_with_id(monkeypatch, client) -> None:
    monkeypatch.setattr(api, "rag_answer", lambda q, d: _fake_answer())
    monkeypatch.setattr(api, "log_conversation", lambda a: "cid-123")
    resp = client.post("/ask", json={"question": "What about MoC?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer_id"] == "cid-123"
    assert data["text"].startswith("Institutions")
    assert data["chunks_used"][0]["para_ids"] == ["82"]
    assert data["citations"][0]["paras"] == "82"


def test_ask_survives_logging_failure(monkeypatch, client) -> None:
    monkeypatch.setattr(api, "rag_answer", lambda q, d: _fake_answer())

    def boom(_a):
        raise RuntimeError("db down")

    monkeypatch.setattr(api, "log_conversation", boom)
    resp = client.post("/ask", json={"question": "q"})
    assert resp.status_code == 200
    assert resp.json()["answer_id"] == ""  # answer still returned


def test_feedback_records(monkeypatch, client) -> None:
    calls: dict = {}
    monkeypatch.setattr(
        api, "log_feedback",
        lambda cid, thumbs, comment=None: calls.update(cid=cid, thumbs=thumbs) or "fid",
    )
    resp = client.post("/feedback", json={"answer_id": "cid-123", "thumbs": "up"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert calls == {"cid": "cid-123", "thumbs": "up"}


def test_feedback_rejects_invalid_thumbs(client) -> None:
    resp = client.post("/feedback", json={"answer_id": "x", "thumbs": "maybe"})
    assert resp.status_code == 422  # Literal["up","down"] validation


def test_health_ok(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "qdrant": True, "postgres": True}


def test_health_degraded(monkeypatch, client) -> None:
    monkeypatch.setattr(api, "ping_postgres", lambda: False)
    assert client.get("/health").json()["status"] == "degraded"
