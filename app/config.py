"""Central configuration for IRB Copilot.

All tunable behaviour (models, retrieval mode, ports, service URLs) is read from
the environment (optionally an `.env` file) via pydantic-settings. Nothing in the
codebase should hardcode a model name, path, or secret — import `settings` here.

See `.env.example` for the documented set of variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]
PromptVersion = Literal["v1", "v2"]
ChunkerKind = Literal["structure", "naive"]

# Repository root (this file is <root>/app/config.py). Relative data paths are
# resolved against it so filesystem access works regardless of the process cwd
# (CLI from root, notebooks from notebooks/, API/UI, containers).
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path_str: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment / `.env`.

    Secrets default to empty so the object can be constructed in tests and at
    import time without leaking or requiring credentials; presence is validated
    at the point of use (see `providers.py`), not here.
    """

    model_config = SettingsConfigDict(
        # Absolute so the root .env loads regardless of cwd (CLI, notebook, API).
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM / embeddings (default provider: OpenAI) ---
    # gpt-4o-mini is the RAG-eval choice (§11.3): ~gpt-4o quality at 1/16th cost.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    # Comma-separated list used by the RAG evaluation (§11.3).
    eval_models: str = Field(default="gpt-4o-mini,gpt-4o", alias="EVAL_MODELS")

    # --- Retrieval / RAG behaviour ---
    # Default is the best config on the DE-BIASED eval (paraphrased questions,
    # §11.2): hybrid_rerank + structure + rewrite off. bm25 wins the naive eval
    # (0.95) but collapses to 0.53 when questions stop echoing the passage, where
    # hybrid_rerank leads (0.64); rewrite hurts on both sets.
    retrieval_mode: RetrievalMode = Field(default="hybrid_rerank", alias="RETRIEVAL_MODE")
    enable_rewrite: bool = Field(default=False, alias="ENABLE_REWRITE")
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

    # Absolute, cwd-independent versions of the path settings (use these for I/O).
    @property
    def data_path(self) -> Path:
        return _resolve(self.data_dir)

    @property
    def raw_path(self) -> Path:
        return _resolve(self.raw_dir)

    @property
    def bm25_index_dir(self) -> Path:
        return _resolve(self.bm25_index_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached `Settings` instance (single source of truth per process)."""
    return Settings()


# Convenience singleton for the common import site.
settings = get_settings()
