"""Evaluation logic tests (SPEC §11): metrics, relevance, sampling, GT parsing."""

from __future__ import annotations

from evaluation.generate_ground_truth import parse_questions
from evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    relevant_by_chunk_id,
    relevant_by_paragraph,
)
from evaluation.sampling import stratified_sample
from ingestion.models import Chunk


# --- metrics ---------------------------------------------------------------- #
def test_hit_rate_counts_any_relevant_in_top_k() -> None:
    rels = [[False, True, False], [False, False, False], [True, False, False]]
    assert hit_rate_at_k(rels, 5) == 2 / 3
    # k truncates: only the first position considered.
    assert hit_rate_at_k(rels, 1) == 1 / 3


def test_mrr_uses_rank_of_first_relevant() -> None:
    rels = [[False, True, False], [True, False, False]]
    # 1/2 + 1/1 over 2 queries.
    assert abs(mrr_at_k(rels, 5) - (0.5 + 1.0) / 2) < 1e-9


def test_metrics_handle_empty() -> None:
    assert hit_rate_at_k([], 5) == 0.0
    assert mrr_at_k([], 5) == 0.0


# --- relevance -------------------------------------------------------------- #
def test_relevant_by_chunk_id() -> None:
    assert relevant_by_chunk_id("abc", "abc")
    assert not relevant_by_chunk_id("abc", "xyz")


def test_relevant_by_paragraph_requires_same_doc_and_overlap() -> None:
    assert relevant_by_paragraph("d1", ["25", "26"], "d1", ["26"])
    assert not relevant_by_paragraph("d1", ["25", "26"], "d1", ["99"])
    assert not relevant_by_paragraph("d2", ["26"], "d1", ["26"])  # wrong doc


# --- stratified sampling ---------------------------------------------------- #
def _chunk(doc_id: str, i: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{i}",
        doc_id=doc_id,
        doc_title=doc_id,
        section_path=[],
        para_ids=[str(i)],
        pages=[1],
        text=f"text {i}",
    )


def test_stratified_sample_is_deterministic_and_covers_docs() -> None:
    chunks = [_chunk("a", i) for i in range(30)] + [_chunk("b", i) for i in range(10)]
    s1 = stratified_sample(chunks, 20, seed=7)
    s2 = stratified_sample(chunks, 20, seed=7)
    assert [c.chunk_id for c in s1] == [c.chunk_id for c in s2]  # reproducible
    assert len(s1) <= 20
    docs = {c.doc_id for c in s1}
    assert docs == {"a", "b"}  # every document represented


def test_stratified_sample_respects_small_n() -> None:
    chunks = [_chunk("a", i) for i in range(5)] + [_chunk("b", i) for i in range(5)]
    assert len(stratified_sample(chunks, 2, seed=1)) <= 2
    assert stratified_sample(chunks, 0, seed=1) == []


# --- ground-truth question parsing ------------------------------------------ #
def test_parse_questions_json_array() -> None:
    text = 'Here: ["What is PD?", "Define LGD.", "Extra?"]'
    assert parse_questions(text, 2) == ["What is PD?", "Define LGD."]


def test_parse_questions_numbered_fallback() -> None:
    text = "1. What is default?\n2) Days past due?\n- Another one"
    assert parse_questions(text, 3) == [
        "What is default?",
        "Days past due?",
        "Another one",
    ]
