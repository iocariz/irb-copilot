"""Model provider factory (SPEC §2, §17).

All model access goes through this module so providers can be swapped via env
vars. Phase 2 needs embeddings for the index stage; chat/completion helpers and
cost accounting are added alongside the RAG flow in a later phase.
"""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from app.config import settings

_EMBED_BATCH = 128


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
    return OpenAI(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts with the configured embedding model (batched)."""
    client = openai_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start : start + _EMBED_BATCH]
        response = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors
