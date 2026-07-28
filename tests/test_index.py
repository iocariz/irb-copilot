"""Tests for the indexing guardrail (SPEC §5 step 4).

`orphan_warning` is pure, so it is tested without a running Qdrant server.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from ingestion import index
from ingestion.index import index_targets, orphan_warning
from ingestion.models import Chunk


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


# --- recreate atomicity: embed before dropping the collection ---------------- #
def test_build_qdrant_does_not_drop_collection_if_embedding_fails(monkeypatch) -> None:
    dropped = {"ensured": False}
    monkeypatch.setattr(index, "QdrantClient", lambda url: object())
    monkeypatch.setattr(
        index, "_ensure_collection",
        lambda *a, **k: dropped.__setitem__("ensured", True),
    )

    def failing_embed(_texts):
        raise RuntimeError("openai unavailable")

    monkeypatch.setattr(index, "embed_texts", failing_embed)
    chunk = Chunk(
        chunk_id="1", doc_id="d", doc_title="D", section_path=[],
        para_ids=["1"], pages=[1], text="t",
    )
    with pytest.raises(RuntimeError):
        index.build_qdrant([chunk], recreate=True, collection="c")
    assert dropped["ensured"] is False  # collection was never (re)created/dropped


# --- BM25 atomic write ------------------------------------------------------- #
def _bm25_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=str(i), doc_id="d", doc_title="D", section_path=[],
            para_ids=[str(i)], pages=[1], text=f"credit risk margin of conservatism {i}",
        )
        for i in range(n)
    ]


def test_build_bm25_writes_full_manifest_and_cleans_temp(tmp_path) -> None:
    out = tmp_path / "bm25_index"
    index.build_bm25(_bm25_chunks(5), bm25_dir=out)
    manifest = out / "chunks.jsonl"
    assert manifest.exists()
    assert sum(1 for line in manifest.open(encoding="utf-8") if line.strip()) == 5
    assert not (tmp_path / ".bm25_index.tmp").exists()  # temp swapped in, not left behind


def test_build_bm25_failure_leaves_existing_index_intact(tmp_path, monkeypatch) -> None:
    import bm25s

    out = tmp_path / "bm25_index"
    index.build_bm25(_bm25_chunks(1), bm25_dir=out)  # establish a good 1-chunk index

    def boom(self, *_a, **_k):  # noqa: ANN001, ANN202
        raise RuntimeError("disk full")

    monkeypatch.setattr(bm25s.BM25, "save", boom)
    with pytest.raises(RuntimeError):
        index.build_bm25(_bm25_chunks(5), bm25_dir=out)  # failing rebuild
    # The live index is untouched (still the original 1 chunk) and temp is gone.
    assert sum(1 for line in (out / "chunks.jsonl").open(encoding="utf-8") if line.strip()) == 1
    assert not (tmp_path / ".bm25_index.tmp").exists()
