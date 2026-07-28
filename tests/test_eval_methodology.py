"""Evaluation-methodology tests: fair relevance, corpus fingerprinting, freshness."""

from __future__ import annotations

import evaluation.generate_ground_truth as gt_mod
from app.config import get_settings
from app.retrieval import RetrievedChunk
from evaluation.corpus import corpus_fingerprint
from evaluation.eval_retrieval import _is_relevant
from evaluation.generate_ground_truth import (
    check_ground_truth_freshness,
    meta_path,
    write_ground_truth_meta,
)
from ingestion.models import Chunk


def _chunk(chunk_id: str, para_ids: list[str], doc_id: str = "d") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, doc_title="D", section_path=[],
        para_ids=para_ids, pages=[1], text="t",
    )


# --- symmetric, fair relevance ---------------------------------------------- #
def test_relevance_is_symmetric_paragraph_overlap() -> None:
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "gt-id"}
    # A different chunk id that still covers the ground-truth paragraph is a hit
    # (was a miss under exact-chunk-id matching — e.g. a hard-split sibling).
    assert _is_relevant(RetrievedChunk(_chunk("other-id", ["82"]), 1.0), row) is True
    # Wrong paragraph -> not relevant.
    assert _is_relevant(RetrievedChunk(_chunk("x", ["99"]), 1.0), row) is False
    # Wrong document -> not relevant even with a matching paragraph number.
    assert _is_relevant(RetrievedChunk(_chunk("x", ["82"], doc_id="other"), 1.0), row) is False


# --- corpus fingerprint ----------------------------------------------------- #
def test_corpus_fingerprint_is_order_independent_and_id_sensitive() -> None:
    a = [_chunk("1", ["1"]), _chunk("2", ["2"])]
    reordered = [_chunk("2", ["2"]), _chunk("1", ["1"])]
    changed = [_chunk("1", ["1"]), _chunk("9", ["2"])]
    assert corpus_fingerprint(a) == corpus_fingerprint(reordered)
    assert corpus_fingerprint(a) != corpus_fingerprint(changed)


# --- ground-truth staleness detection --------------------------------------- #
def test_ground_truth_freshness_detects_corpus_drift(tmp_path, monkeypatch) -> None:
    chunks = [_chunk("1", ["1"]), _chunk("2", ["2"])]
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text("question,doc_id,chunk_id,para_ids\n", encoding="utf-8")
    write_ground_truth_meta(gt_path, chunks)
    assert meta_path(gt_path).exists()

    # Same corpus -> no warning.
    monkeypatch.setattr(gt_mod, "load_chunks", lambda kind, s=None: chunks)
    assert check_ground_truth_freshness(gt_path, get_settings()) is None

    # Re-chunked corpus (an id changed) -> loud warning.
    drifted = [_chunk("1", ["1"]), _chunk("Z", ["2"])]
    monkeypatch.setattr(gt_mod, "load_chunks", lambda kind, s=None: drifted)
    warning = check_ground_truth_freshness(gt_path, get_settings())
    assert warning is not None and "regenerate" in warning


def test_ground_truth_freshness_no_sidecar_is_silent(tmp_path) -> None:
    gt_path = tmp_path / "ground_truth.csv"
    gt_path.write_text("question,doc_id,chunk_id,para_ids\n", encoding="utf-8")
    # No .meta.json sidecar (older ground truth) -> no warning, no corpus load.
    assert check_ground_truth_freshness(gt_path, get_settings()) is None
