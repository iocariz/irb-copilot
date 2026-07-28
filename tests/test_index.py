"""Tests for the indexing guardrail (SPEC §5 step 4).

`orphan_warning` is pure, so it is tested without a running Qdrant server.
"""

from __future__ import annotations

from app.config import get_settings
from ingestion.index import index_targets, orphan_warning


def test_no_warning_when_counts_match() -> None:
    assert orphan_warning(1226, 1226) is None


def test_no_warning_when_collection_smaller() -> None:
    # Partial index (e.g. interrupted) is not an orphan situation.
    assert orphan_warning(500, 1226) is None


def test_warning_reports_orphan_count() -> None:
    msg = orphan_warning(3656, 1226)
    assert msg is not None
    assert "2430 orphaned" in msg
    assert "--recreate" in msg


# --- index target routing (naive must not touch production) ------------------ #
def test_index_targets_structure_is_production() -> None:
    s = get_settings()
    collection, bm25_dir = index_targets("structure")
    assert collection == s.qdrant_collection
    assert bm25_dir == s.bm25_index_dir


def test_index_targets_naive_is_isolated_from_production() -> None:
    s = get_settings()
    collection, bm25_dir = index_targets("naive")
    assert collection == f"{s.qdrant_collection}_naive"
    assert bm25_dir == s.data_path / "bm25_index_naive"
    assert collection != s.qdrant_collection
    assert bm25_dir != s.bm25_index_dir
