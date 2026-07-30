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

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RetrievalMode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]
PromptVersion = Literal["v1", "v2"]
ChunkerKind = Literal["structure", "naive"]
# How much query rewriting to apply before retrieval (SPEC §7). Three levels,
# because the two steps the spec describes have very different cost and effect:
#   off      — retrieve on the user's question verbatim
#   glossary — deterministic acronym expansion only (no LLM call, no cost)
#   llm      — glossary expansion plus an LLM reformulation
# Replaces the old boolean ENABLE_REWRITE, which conflated "no LLM rewrite" with
# "no rewriting at all": glossary expansion ran unconditionally, so the shipped
# `false` setting was really this `glossary` level and was never evaluated.
RewriteMode = Literal["off", "glossary", "llm"]

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
    # gpt-4o-mini is the answer model: cheapest option that does this task well
    # (grounded extraction with a citation format, not reasoning). Re-confirm via
    # `make eval-rag` — see the artifact-status note in the README.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")
    # Comma-separated list used by the RAG evaluation (§11.3).
    eval_models: str = Field(default="gpt-4o-mini,gpt-4o", alias="EVAL_MODELS")
    # LLM-as-a-judge model (§11.3). gpt-5.4-mini rather than gpt-4o: it is a
    # different family from the gpt-4o-mini answers (so no self-preference bias),
    # accepts temperature=0 (gpt-5-mini does not — reproducibility), emits no
    # reasoning tokens by default, is ~3x cheaper than gpt-4o, and its much larger
    # TPM headroom (200k vs gpt-4o's 30k here) removes the eval rate-limit wall.
    judge_model: str = Field(default="gpt-5.4-mini", alias="JUDGE_MODEL")

    # --- Retrieval / RAG behaviour ---
    # Best config on the DE-BIASED eval (§11.2, 800 questions): hybrid_rerank +
    # structure + no rewriting, hit@5 0.7425. bm25 wins the STANDARD set (0.921 vs
    # 0.915) but collapses to 0.585 once questions stop echoing the passage, where
    # hybrid_rerank leads at 0.743 — the rank order fully reverses. Rewriting hurts
    # on both sets, monotonically (off >= glossary >= llm in 8/8 configs).
    retrieval_mode: RetrievalMode = Field(default="hybrid_rerank", alias="RETRIEVAL_MODE")
    # `off` is evidence-based: measured conditionally on the queries where
    # glossary expansion actually changes something (the 200 ground-truth
    # questions containing an acronym), it HURT — bm25 hit@5 -0.035 / mrr -0.071,
    # vector hit@5 -0.050 / mrr -0.040. It appends "LGD (loss given default)",
    # which dilutes BM25 term weights and shifts the embedding away from the
    # corpus's own vocabulary — and the corpus uses the acronyms too.
    # This ran unconditionally before (ENABLE_REWRITE only gated the LLM step),
    # so it had never been evaluated. See task.md T2.
    rewrite_mode: RewriteMode = Field(default="off", alias="REWRITE_MODE")
    # DEPRECATED, read only to migrate old deployments — see the validator below.
    # Declared as a real field so pydantic-settings collects it from the
    # environment *and* from `.env`; excluded from dumps so it is not mistaken
    # for live configuration. Use `rewrite_mode`.
    enable_rewrite: bool | None = Field(default=None, alias="ENABLE_REWRITE", exclude=True)
    # Rewrite a follow-up into a standalone query (using chat history) before
    # retrieval, so references like "the next paragraph" search well. Only fires
    # when history is present; costs one cheap LLM call per follow-up.
    condense_history: bool = Field(default=True, alias="CONDENSE_HISTORY")
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

    # --- API protection (matters once the API is publicly reachable) ---
    # If set, /ask, /ask/stream and /feedback require this key (X-API-Key header).
    api_key: str = Field(default="", alias="API_KEY")
    # Fail fast at startup if auth is required but no key is set (set true in prod
    # so write endpoints are never served unauthenticated by accident).
    require_api_key: bool = Field(default=False, alias="REQUIRE_API_KEY")
    # Expose the interactive API docs (/docs, /redoc, /openapi.json). Disable in
    # production so the full API schema isn't publicly browsable.
    docs_enabled: bool = Field(default=True, alias="DOCS_ENABLED")
    # Per-client (IP) requests/minute for those endpoints; 0 disables.
    rate_limit_per_minute: int = Field(default=30, alias="RATE_LIMIT_PER_MINUTE")
    # Number of trusted reverse proxies in front of the app (Caddy = 1). The real
    # client IP is read this many hops from the RIGHT of X-Forwarded-For, so
    # clients can't spoof it via the leftmost hop. Set 0 when there is no proxy.
    trusted_proxy_hops: int = Field(default=1, alias="TRUSTED_PROXY_HOPS")
    # Input bounds to cap prompt cost / abuse.
    max_question_chars: int = Field(default=2000, alias="MAX_QUESTION_CHARS")
    max_history_messages: int = Field(default=10, alias="MAX_HISTORY_MESSAGES")
    max_doc_ids: int = Field(default=20, alias="MAX_DOC_IDS")
    # Hard cap on answer-generation output tokens, so one request can't run the
    # model out to the full context window (cost / abuse guard). The rewrite and
    # condense calls are already bounded separately.
    max_answer_tokens: int = Field(default=1500, alias="MAX_ANSWER_TOKENS")

    # --- Local paths ---
    data_dir: str = Field(default="data", alias="DATA_DIR")
    raw_dir: str = Field(default="data/raw", alias="RAW_DIR")
    bm25_index_path: str = Field(default="data/bm25_index", alias="BM25_INDEX_PATH")

    @model_validator(mode="after")
    def _migrate_enable_rewrite(self) -> Settings:
        """Map a legacy ``ENABLE_REWRITE`` onto ``REWRITE_MODE``.

        An existing deployment may still set the old boolean; honour it rather
        than silently changing that deployment's retrieval behaviour. The mapping
        is what the flag actually did — ``false`` still ran glossary expansion
        (never "no rewriting"), ``true`` added the LLM call.

        An explicitly-set ``REWRITE_MODE`` always wins: `model_fields_set` covers
        values from the environment and from `.env`, not just literal kwargs.
        """
        if self.enable_rewrite is not None and "rewrite_mode" not in self.model_fields_set:
            self.rewrite_mode = "llm" if self.enable_rewrite else "glossary"
        return self

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
