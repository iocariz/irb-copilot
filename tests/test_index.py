"""Tests for the indexing guardrail (SPEC §5 step 4).

`orphan_warning` is pure, so it is tested without a running Qdrant server.
"""

from __future__ import annotations

from ingestion.index import orphan_warning


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
