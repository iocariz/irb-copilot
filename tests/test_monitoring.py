"""Monitoring tests (SPEC §12): traffic provenance and dashboard scoping.

The RAG evaluation seeds hundreds of synthetic conversations so the Grafana judge
panel has data. Before the `source` column existed they were indistinguishable
from real usage: 1,901 of 1,921 rows were evaluation traffic, so every usage,
cost and latency panel was reporting the evaluation's own activity as though users
had produced it. These tests keep the distinction and the panel filters honest.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from monitoring.db import _ADDED_COLUMNS, EVAL, LIVE, Conversation, log_conversation

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "monitoring/grafana/provisioning/dashboards/irb_copilot.json"
SCHEMA = ROOT / "monitoring/schema.sql"

# Panels that describe real usage; the evaluation's rows must not appear in them.
_LIVE_PANELS = {
    "Questions per day",
    "Thumbs up/down ratio",
    "Cost over time (USD)",
    "Latency p50 / p95 (ms)",
    "Retrieval mode usage",
    "Ungrounded-citation rate (hallucinations)",
    "Truncated answers (hit the token cap)",
}
# Panels that exist *because* of the evaluation.
_EVAL_PANELS = {"Judge relevance distribution"}


@pytest.fixture(scope="module")
def panels() -> dict[str, str]:
    dash = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    return {p["title"]: " ".join(p["targets"][0]["rawSql"].split()) for p in dash["panels"]}


# --- provenance -------------------------------------------------------------- #
def test_log_conversation_defaults_to_live() -> None:
    """A real /ask call must not have to remember to say so."""
    assert inspect.signature(log_conversation).parameters["source"].default == LIVE


def test_source_values_are_distinct_and_short() -> None:
    assert LIVE != EVAL
    # The column is String(8); longer values would be silently truncated.
    assert len(LIVE) <= 8 and len(EVAL) <= 8


def test_conversation_model_has_provenance_and_truncation_columns() -> None:
    columns = Conversation.__table__.columns
    assert columns["source"].nullable is False, "provenance must never be unknown"
    assert columns["source"].server_default is not None, "existing rows need a default"
    assert columns["answer_truncated"].nullable is True  # unknown for older rows


def test_migration_adds_both_columns_idempotently() -> None:
    """init_db runs on every startup, so the statements must be re-runnable."""
    joined = " ".join(_ADDED_COLUMNS)
    assert "ADD COLUMN IF NOT EXISTS source" in joined
    assert "ADD COLUMN IF NOT EXISTS answer_truncated" in joined
    assert f"DEFAULT '{LIVE}'" in joined, "back-fill must not orphan existing rows"


def test_schema_sql_matches_the_model() -> None:
    """schema.sql initialises a fresh container DB; it must not drift."""
    sql = SCHEMA.read_text(encoding="utf-8")
    assert "source" in sql and f"DEFAULT '{LIVE}'" in sql
    assert "answer_truncated" in sql
    # Every usage panel filters on source, so it should be indexed.
    assert "idx_conversations_source_ts" in sql


# --- dashboard scoping ------------------------------------------------------- #
@pytest.mark.parametrize("title", sorted(_LIVE_PANELS))
def test_usage_panels_exclude_evaluation_traffic(panels, title) -> None:
    assert title in panels, f"panel {title!r} missing from the dashboard"
    assert "source = 'live'" in panels[title], (
        f"{title!r} would report the evaluation's synthetic conversations as usage"
    )


@pytest.mark.parametrize("title", sorted(_EVAL_PANELS))
def test_judge_panel_reads_evaluation_traffic(panels, title) -> None:
    assert "source = 'eval'" in panels[title]


def test_every_panel_states_its_scope(panels) -> None:
    """No panel may silently mix the two populations."""
    unscoped = [t for t, sql in panels.items() if "source = '" not in sql]
    assert not unscoped, f"panels with no source filter: {unscoped}"


def test_dashboard_still_satisfies_the_five_panel_requirement(panels) -> None:
    assert len(panels) >= 5
    assert set(panels) == _LIVE_PANELS | _EVAL_PANELS
