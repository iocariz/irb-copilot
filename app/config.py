"""Central configuration for IRB Copilot.

All tunable behaviour (models, retrieval mode, ports, service URLs) is read from
the environment (optionally an `.env` file) via pydantic-settings. Nothing in the
codebase should hardcode a model name, path, or secret — import `settings` here.

See `.env.example` for the documented set of variables.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]
PromptVersion = Literal["v1", "v2"]
ChunkerKind = Literal["structure", "naive"]


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment / `.env`.

    Secrets default to empty so the object can be constructed in tests and at
    import time without leaking or requiring credentials; presence is validated
    at the point of use (see `providers.py`), not here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM / embeddings (default provider: OpenAI) ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    # Comma-separated list used by the RAG evaluation (§11.3).
    eval_models: str = Field(default="gpt-4o-mini,gpt-4o", alias="EVAL_MODELS")

    # --- Retrieval / RAG behaviour ---
    retrieval_mode: RetrievalMode = Field(default="hybrid", alias="RETRIEVAL_MODE")
    enable_rewrite: bool = Field(default=True, alias="ENABLE_REWRITE")
    prompt_version: PromptVersion = Field(default="v2", alias="PROMPT_VERSION")
    chunker: ChunkerKind = Field(default="structure", alias="CHUNKER")
    top_k: int = Field(default=5, alias="TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")

    # --- Vector store (Qdrant) ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="irb_chunks", alias="QDRANT_COLLECTION")

    # --- Monitoring (Postgres) ---
    postgres_dsn: str = Field(
        default="postgresql+psycopg://irb:irb@localhost:5432/irb",
        alias="POSTGRES_DSN",
    )

    # --- Service wiring ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_url: str = Field(default="http://localhost:8000", alias="API_URL")
    ui_port: int = Field(default=8501, alias="UI_PORT")
    grafana_port: int = Field(default=3000, alias="GRAFANA_PORT")

    # --- Local paths ---
    data_dir: str = Field(default="data", alias="DATA_DIR")
    raw_dir: str = Field(default="data/raw", alias="RAW_DIR")
    bm25_index_path: str = Field(default="data/bm25_index", alias="BM25_INDEX_PATH")

    @property
    def eval_models_list(self) -> list[str]:
        """`EVAL_MODELS` parsed into a clean list."""
        return [m.strip() for m in self.eval_models.split(",") if m.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance (single source of truth per process)."""
    return Settings()


# Convenience singleton for the common import site.
settings = get_settings()
