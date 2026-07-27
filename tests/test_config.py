"""Phase 1 smoke tests for `app.config`.

These prove the configuration surface loads with sane defaults and honours the
environment, without requiring any secrets or running services.
"""

from __future__ import annotations

import importlib

from app.config import Settings


def test_defaults_match_spec() -> None:
    """Defaults reflect the documented, evaluation-picked defaults (SPEC §13)."""
    s = Settings(_env_file=None)
    assert s.llm_model == "gpt-4o-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.retrieval_mode == "bm25"  # eval-chosen default (§11.2)
    assert s.enable_rewrite is False
    assert s.prompt_version == "v2"
    assert s.chunker == "structure"
    assert s.top_k == 5
    assert s.qdrant_collection == "irb_chunks"


def test_secret_is_optional_for_import_and_tests() -> None:
    """OPENAI_API_KEY must not be required to construct Settings (SPEC §17)."""
    assert Settings(_env_file=None).openai_api_key == ""


def test_env_overrides_are_read(monkeypatch) -> None:
    """Environment variables (case-insensitive aliases) override defaults."""
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    monkeypatch.setenv("ENABLE_REWRITE", "false")
    monkeypatch.setenv("TOP_K", "10")
    s = Settings(_env_file=None)
    assert s.retrieval_mode == "bm25"
    assert s.enable_rewrite is False
    assert s.top_k == 10


def test_eval_models_list_parsing() -> None:
    s = Settings(_env_file=None, EVAL_MODELS="gpt-4o-mini, gpt-4o ,")
    assert s.eval_models_list == ["gpt-4o-mini", "gpt-4o"]


def test_data_paths_are_absolute_and_root_anchored() -> None:
    """Relative path settings resolve against the repo root, not the cwd."""
    from app.config import PROJECT_ROOT

    s = Settings(_env_file=None)
    assert s.bm25_index_dir.is_absolute()
    assert s.bm25_index_dir == PROJECT_ROOT / "data" / "bm25_index"
    assert s.data_path == PROJECT_ROOT / "data"
    assert s.raw_path == PROJECT_ROOT / "data" / "raw"


def test_absolute_path_setting_is_left_unchanged() -> None:
    s = Settings(_env_file=None, BM25_INDEX_PATH="/tmp/idx")
    assert str(s.bm25_index_dir) == "/tmp/idx"


def test_module_exposes_cached_singleton() -> None:
    """`settings` singleton and `get_settings()` cache are wired up."""
    import app.config as cfg

    importlib.reload(cfg)
    assert cfg.get_settings() is cfg.get_settings()
    assert cfg.settings is cfg.get_settings()
