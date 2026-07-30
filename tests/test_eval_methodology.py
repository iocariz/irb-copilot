"""Evaluation-methodology tests: fair relevance, corpus fingerprinting, freshness."""

from __future__ import annotations

import evaluation.generate_ground_truth as gt_mod
from app.config import get_settings
from app.retrieval import RetrievedChunk
from evaluation.corpus import corpus_fingerprint
from evaluation.eval_retrieval import GroundTruthIndex, _is_relevant
from evaluation.generate_ground_truth import (
    check_ground_truth_freshness,
    meta_path,
    write_ground_truth_meta,
)
from ingestion.models import Chunk


def _chunk(
    chunk_id: str, para_ids: list[str], doc_id: str = "d", text: str = "t"
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id=doc_id, doc_title="D", section_path=[],
        para_ids=para_ids, pages=[1], text=text,
    )


# --- relevance is pinned to the passage, not the paragraph number ----------- #
_GT_TEXT = (
    "Institutions shall apply an appropriate margin of conservatism to account for "
    "estimation error and any deficiency in the data used to estimate risk parameters."
)
_OTHER_TEXT = (
    "Competent authorities should assess whether the rating system captures all "
    "relevant obligor characteristics before granting permission for its use."
)


def _gt_index(*chunks: Chunk) -> GroundTruthIndex:
    return GroundTruthIndex(chunks)


def test_exact_chunk_id_is_always_relevant() -> None:
    gt_chunk = _chunk("gt-id", ["82"], text=_GT_TEXT)
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "gt-id"}
    hit = RetrievedChunk(gt_chunk, 1.0)
    assert _is_relevant(hit, row, _gt_index(gt_chunk)) is True


def test_same_paragraph_number_different_passage_is_a_miss() -> None:
    """The core fix: paragraph numbers restart per section, so a chunk that
    merely shares the number — but not the text — must not count as a hit."""
    gt_chunk = _chunk("gt-id", ["82"], text=_GT_TEXT)
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "gt-id"}
    impostor = _chunk("other-id", ["82"], text=_OTHER_TEXT)  # same number, other section
    assert _is_relevant(RetrievedChunk(impostor, 1.0), row, _gt_index(gt_chunk)) is False


def test_chunk_containing_the_ground_truth_passage_is_a_hit() -> None:
    """A differently-merged chunk that still contains the passage counts —
    this is what keeps the structure/naive comparison fair."""
    gt_chunk = _chunk("gt-id", ["82"], text=_GT_TEXT)
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "gt-id"}
    # Merged with a neighbouring paragraph, and no paragraph id in common.
    wider = _chunk("merged-id", ["81", "82"], text=f"{_OTHER_TEXT}\n\n{_GT_TEXT}")
    assert _is_relevant(RetrievedChunk(wider, 1.0), row, _gt_index(gt_chunk)) is True


def test_wrong_document_is_a_miss_even_with_identical_text() -> None:
    gt_chunk = _chunk("gt-id", ["82"], text=_GT_TEXT)
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "gt-id"}
    other_doc = _chunk("x", ["82"], doc_id="other", text=_GT_TEXT)
    assert _is_relevant(RetrievedChunk(other_doc, 1.0), row, _gt_index(gt_chunk)) is False


def test_stale_ground_truth_falls_back_and_is_reported() -> None:
    """A ground-truth chunk missing from the corpus has no text to compare, so
    the loose rule is used — but the run must say so."""
    row = {"doc_id": "d", "para_ids_list": ["82"], "chunk_id": "vanished"}
    index = _gt_index(_chunk("still-here", ["1"], text=_GT_TEXT))
    assert _is_relevant(RetrievedChunk(_chunk("x", ["82"]), 1.0), row, index) is True
    assert "vanished" in index.missing


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
