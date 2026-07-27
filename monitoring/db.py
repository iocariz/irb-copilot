"""Monitoring persistence (SPEC §12).

SQLAlchemy models for `conversations` and `feedback`, plus logging helpers used
by the API. The engine is created lazily so importing this module never opens a
connection (safe for tests and for a DB that isn't up yet).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    question: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(8))
    retrieval_mode: Mapped[str] = mapped_column(String(32))
    chunk_ids: Mapped[list] = mapped_column(JSON)
    tokens_in: Mapped[int] = mapped_column(Integer)
    tokens_out: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[int] = mapped_column(Integer)
    judge_relevance: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id")
    )
    thumbs: Mapped[str] = mapped_column(String(8))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))


@lru_cache(maxsize=1)
def _engine():  # noqa: ANN202
    return create_engine(get_settings().postgres_dsn, pool_pre_ping=True)


@lru_cache(maxsize=1)
def _sessions():  # noqa: ANN202
    return sessionmaker(bind=_engine())


def init_db() -> None:
    """Create tables if they do not exist (idempotent)."""
    Base.metadata.create_all(_engine())


def log_conversation(answer) -> str:  # noqa: ANN001 — app.rag.Answer (avoid import cycle)
    """Persist an answered question; return the new conversation id."""
    conversation_id = str(uuid.uuid4())
    with _sessions().begin() as session:
        session.add(
            Conversation(
                id=conversation_id,
                ts=_now(),
                question=answer.question,
                rewritten_query=answer.rewritten_query,
                answer=answer.text,
                model=answer.model,
                prompt_version=answer.prompt_version,
                retrieval_mode=answer.retrieval_mode,
                chunk_ids=[c.chunk_id for c in answer.chunks_used],
                tokens_in=answer.tokens_in,
                tokens_out=answer.tokens_out,
                cost_usd=answer.cost_usd,
                latency_ms=answer.latency_ms,
                judge_relevance=None,
            )
        )
    return conversation_id


def log_feedback(conversation_id: str, thumbs: str, comment: str | None = None) -> str:
    """Persist a thumbs up/down (with optional comment); return feedback id."""
    feedback_id = str(uuid.uuid4())
    with _sessions().begin() as session:
        session.add(
            Feedback(
                id=feedback_id,
                conversation_id=conversation_id,
                thumbs=thumbs,
                comment=comment,
                ts=_now(),
            )
        )
    return feedback_id


def ping_postgres() -> bool:
    """Return True if the database answers a trivial query."""
    try:
        with _engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 — health check must not raise.
        return False


def _now() -> datetime:
    return datetime.now(UTC)
