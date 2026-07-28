"""FastAPI backend (SPEC §9).

Endpoints:
* POST /ask       -> run the RAG flow, log the conversation, return the Answer + id
* POST /feedback  -> record a thumbs up/down for a prior answer
* GET  /health    -> report Qdrant + Postgres connectivity

Conversation logging is best-effort: a monitoring outage must not stop answers.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.config import get_settings
from app.rag import Answer, answer_stream
from app.rag import answer as rag_answer
from app.retrieval import get_retriever
from app.security import RateLimiter, client_ip
from monitoring.db import init_db, log_conversation, log_feedback, ping_postgres

_settings = get_settings()
_limiter = RateLimiter(_settings.rate_limit_per_minute)


def guard(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Auth (optional API key) + per-IP rate limiting for the write endpoints."""
    if _settings.api_key and x_api_key != _settings.api_key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    ip = client_ip(
        request.headers.get("x-forwarded-for"),
        request.client.host if request.client else "unknown",
    )
    if not _limiter.allow(ip, time.time()):
        raise HTTPException(status_code=429, detail="rate limit exceeded — slow down")


class HistoryMessage(BaseModel):
    """A prior conversation turn. Roles are constrained to user/assistant so a
    client cannot inject a `system` (or tool) message via the history field."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def _bound_content(cls, value: str) -> str:
        limit = get_settings().max_question_chars * 2
        if len(value) > limit:
            raise ValueError(f"history message exceeds {limit} characters")
        return value


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] | None = None
    history: list[HistoryMessage] | None = None

    @field_validator("question")
    @classmethod
    def _bound_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        limit = get_settings().max_question_chars
        if len(value) > limit:
            raise ValueError(f"question exceeds {limit} characters")
        return value

    @field_validator("doc_ids")
    @classmethod
    def _bound_doc_ids(cls, value: list[str] | None) -> list[str] | None:
        if value and len(value) > get_settings().max_doc_ids:
            raise ValueError("too many doc_ids")
        return value

    @field_validator("history")
    @classmethod
    def _bound_history(cls, value: list | None) -> list | None:
        if value and len(value) > get_settings().max_history_messages:
            raise ValueError("history too long")
        return value


class AskResponse(Answer):
    """The full Answer plus the id needed to attach feedback."""

    answer_id: str


class FeedbackRequest(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    thumbs: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=2000)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 — start even if DB is briefly down.
        print(f"[api] init_db failed (monitoring may be unavailable): {exc}")
    try:
        get_retriever().warm()  # load index/model once now, not on the first request
    except Exception as exc:  # noqa: BLE001 — e.g. corpus not ingested yet.
        print(f"[api] retriever warm-up skipped: {exc}")
    yield


app = FastAPI(title="IRB Copilot", version="0.1.0", lifespan=lifespan)


def _history_dicts(request: AskRequest) -> list[dict[str, str]] | None:
    return [m.model_dump() for m in request.history] if request.history else None


_UPSTREAM_ERROR = "failed to generate an answer (retrieval or model error)"


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(guard)])
def ask(request: AskRequest) -> AskResponse:
    try:
        answer = rag_answer(request.question, request.doc_ids, history=_history_dicts(request))
    except Exception as exc:  # noqa: BLE001 — turn provider/retrieval faults into a clean 502.
        print(f"[api] /ask failed: {exc!r}")
        raise HTTPException(status_code=502, detail=_UPSTREAM_ERROR) from exc
    answer_id = _log(answer)
    return AskResponse(answer_id=answer_id, **answer.model_dump())


@app.post("/ask/stream", dependencies=[Depends(guard)])
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: `sources` once, then `token` deltas, then `done`
    (the full Answer + answer_id) — or an `error` event if generation fails."""

    def events():  # noqa: ANN202
        final: Answer | None = None
        try:
            for kind, payload in answer_stream(
                request.question, request.doc_ids, history=_history_dicts(request)
            ):
                if kind == "sources":
                    yield _sse("sources", [s.model_dump() for s in payload])
                elif kind == "token":
                    yield _sse("token", {"text": payload})
                elif kind == "answer":
                    final = payload
        except Exception as exc:  # noqa: BLE001 — the stream is already 200; signal via SSE.
            print(f"[api] /ask/stream failed: {exc!r}")
            yield _sse("error", {"detail": _UPSTREAM_ERROR})
            return
        answer_id = _log(final) if final else ""
        yield _sse("done", {"answer_id": answer_id, **(final.model_dump() if final else {})})

    return StreamingResponse(events(), media_type="text/event-stream")


def _log(answer: Answer | None) -> str:
    """Best-effort conversation logging; answering must survive DB errors."""
    if answer is None:
        return ""
    try:
        return log_conversation(answer)
    except Exception as exc:  # noqa: BLE001
        print(f"[api] log_conversation failed: {exc}")
        return ""


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/feedback", dependencies=[Depends(guard)])
def feedback(request: FeedbackRequest) -> dict[str, str]:
    try:
        feedback_id = log_feedback(request.answer_id, request.thumbs, request.comment)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"feedback failed: {exc}") from exc
    return {"status": "ok", "feedback_id": feedback_id}


@app.get("/health")
def health() -> dict[str, object]:
    qdrant_ok = ping_qdrant()
    postgres_ok = ping_postgres()
    return {
        "status": "ok" if (qdrant_ok and postgres_ok) else "degraded",
        "qdrant": qdrant_ok,
        "postgres": postgres_ok,
    }


def ping_qdrant() -> bool:
    """Return True if Qdrant answers a collections listing."""
    try:
        from qdrant_client import QdrantClient

        QdrantClient(url=get_settings().qdrant_url).get_collections()
        return True
    except Exception:  # noqa: BLE001 — health check must not raise.
        return False
