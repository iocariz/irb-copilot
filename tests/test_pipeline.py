"""Tests for ingestion stage selection (shared by the CLI and the Prefect flow)."""

from __future__ import annotations

import pytest

from ingestion import pipeline
from ingestion.models import Chunk
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


def _write_chunk(chunks_dir, doc_id: str, chunk_id: str, kind: str = "structure") -> None:
    chunk = Chunk(
        chunk_id=chunk_id, doc_id=doc_id, doc_title=doc_id, section_path=[],
        para_ids=["1"], pages=[1], text="text",
    )
    (chunks_dir / f"{doc_id}.{kind}.jsonl").write_text(chunk.model_dump_json() + "\n")


def test_run_index_uses_full_corpus_not_a_subset(tmp_path, monkeypatch) -> None:
    # Regression: --only-doc must not shrink the (monolithic) BM25 index. run_index
    # always reads every chunk file, so both documents are indexed.
    _write_chunk(tmp_path, "docA", "a1")
    _write_chunk(tmp_path, "docB", "b1")
    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "index_chunks",
        lambda chunks, recreate=False, collection=None, bm25_dir=None: captured.update(
            ids={c.chunk_id for c in chunks}, recreate=recreate, collection=collection
        ),
    )
    pipeline.run_index(tmp_path, kind="structure", recreate=True)
    assert captured["ids"] == {"a1", "b1"}
    assert captured["recreate"] is True
    from app.config import get_settings
    assert captured["collection"] == get_settings().qdrant_collection  # production


def test_run_index_naive_targets_isolated_namespace(tmp_path, monkeypatch) -> None:
    # --chunker naive must NOT write into the production collection.
    _write_chunk(tmp_path, "docA", "a1", kind="naive")
    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "index_chunks",
        lambda chunks, recreate=False, collection=None, bm25_dir=None: captured.update(
            collection=collection
        ),
    )
    pipeline.run_index(tmp_path, kind="naive", recreate=True)
    from app.config import get_settings
    assert captured["collection"] == f"{get_settings().qdrant_collection}_naive"


def test_run_index_errors_without_chunk_files(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pipeline.run_index(tmp_path, kind="structure", recreate=False)
