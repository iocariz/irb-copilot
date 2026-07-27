"""Tests for ingestion stage selection (shared by the CLI and the Prefect flow)."""

from __future__ import annotations

import pytest

from ingestion.pipeline import STAGES, active_stages


def test_active_stages_full_pipeline() -> None:
    assert active_stages("download") == STAGES


def test_active_stages_resume_from_middle() -> None:
    assert active_stages("parse") == ("parse", "chunk", "index")


def test_active_stages_bounded_range() -> None:
    assert active_stages("parse", "chunk") == ("parse", "chunk")


def test_active_stages_single() -> None:
    assert active_stages("chunk", "chunk") == ("chunk",)


def test_active_stages_rejects_inverted_range() -> None:
    with pytest.raises(ValueError):
        active_stages("index", "parse")
