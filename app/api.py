"""FastAPI backend (SPEC §9).

Endpoints:
* POST /ask       -> run the RAG flow, log the conversation, return the Answer + id
* POST /feedback  -> record a thumbs up/down for a prior answer
* GET  /health    -> report Qdrant + Postgres connectivity

Conversation logging is best-effort: a monitoring outage must not stop answers.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.rag import Answer, answer_stream
from app.rag import answer as rag_answer
from monitoring.db import init_db, log_conversation, log_feedback, ping_postgres


class AskRequest(BaseModel):
    question: str
    doc_ids: list[str] | None = None
    # Prior turns [{role, content}] for follow-up questions (optional).
    history: list[dict[str, str]] | None = None


class AskResponse(Answer):
    """The full Answer plus the id needed to attach feedback."""

    answer_id: str


class FeedbackRequest(BaseModel):
    answer_id: str
    thumbs: Literal["up", "down"]
    comment: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 — start even if DB is briefly down.
        print(f"[api] init_db failed (monitoring may be unavailable): {exc}")
    yield


app = FastAPI(title="IRB Copilot", version="0.1.0", lifespan=lifespan)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    answer = rag_answer(request.question, request.doc_ids, history=request.history)
    answer_id = _log(answer)
    return AskResponse(answer_id=answer_id, **answer.model_dump())


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    """Server-sent events: `sources` once, then `token` deltas, then `done`
    (the full Answer + answer_id)."""

    def events():  # noqa: ANN202
        final: Answer | None = None
        for kind, payload in answer_stream(
            request.question, request.doc_ids, history=request.history
        ):
            if kind == "sources":
                yield _sse("sources", [s.model_dump() for s in payload])
            elif kind == "token":
                yield _sse("token", {"text": payload})
            elif kind == "answer":
                final = payload
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


@app.post("/feedback")
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
