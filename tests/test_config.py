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
    assert s.retrieval_mode == "hybrid_rerank"  # de-biased eval winner (§11.2)
    assert s.rewrite_mode == "off"  # evaluated winner (§11.2); see task.md T2
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
    monkeypatch.setenv("REWRITE_MODE", "off")
    monkeypatch.setenv("TOP_K", "10")
    s = Settings(_env_file=None)
    assert s.retrieval_mode == "bm25"
    assert s.rewrite_mode == "off"
    assert s.top_k == 10


# --- legacy ENABLE_REWRITE migration ---------------------------------------- #
def test_legacy_enable_rewrite_false_maps_to_glossary(monkeypatch) -> None:
    """The old flag never meant "no rewriting": glossary expansion always ran.
    A deployment still setting it must keep the behaviour it had."""
    monkeypatch.setenv("ENABLE_REWRITE", "false")
    assert Settings(_env_file=None).rewrite_mode == "glossary"


def test_legacy_enable_rewrite_true_maps_to_llm(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_REWRITE", "true")
    assert Settings(_env_file=None).rewrite_mode == "llm"


def test_explicit_rewrite_mode_wins_over_legacy_flag(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_REWRITE", "true")
    monkeypatch.setenv("REWRITE_MODE", "off")
    assert Settings(_env_file=None).rewrite_mode == "off"


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


# --- .env.example must be safe to copy (found by a clean-clone run) ---------- #
def test_env_example_has_no_value_that_is_actually_a_comment() -> None:
    """`make setup` copies .env.example to .env, so a parsing quirk there breaks
    every fresh clone.

    dotenv reads an inline comment that follows an *empty* value as the value:
    `API_KEY=    # if set, ...` gave `api_key='# if set, ...'`, so every /ask and
    /feedback returned 401 with a key nobody knows. Put such comments on their own
    line above the variable.
    """
    from pathlib import Path

    from dotenv import dotenv_values

    example = Path(__file__).resolve().parent.parent / ".env.example"
    offenders = {
        key: value
        for key, value in dotenv_values(example).items()
        if value and value.lstrip().startswith("#")
    }
    assert not offenders, (
        f"these .env.example values are comments, not settings: {offenders}"
    )


def test_settings_built_from_env_example_serve_requests_unauthenticated() -> None:
    """The shipped example must not silently enable API-key auth."""
    from pathlib import Path

    example = Path(__file__).resolve().parent.parent / ".env.example"
    settings = Settings(_env_file=str(example))
    assert settings.api_key == "", "a fresh clone would 401 on every request"
    assert settings.require_api_key is False
