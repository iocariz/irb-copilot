"""Evaluation logic tests (SPEC §11): metrics, relevance, sampling, GT parsing."""

from __future__ import annotations

from pathlib import Path

from evaluation.eval_retrieval import output_suffix
from evaluation.generate_ground_truth import parse_questions
from evaluation.metrics import (
    hit_rate_at_k,
    lexical_overlap,
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
# --- lexical overlap (de-biasing signal) + output suffix --------------------- #
def test_lexical_overlap_high_when_question_reuses_vocabulary() -> None:
    passage = "margin of conservatism in LGD estimation for defaulted exposures"
    question = "What margin of conservatism applies to LGD estimation?"
    assert lexical_overlap(question, passage) > 0.6


def test_lexical_overlap_low_when_paraphrased() -> None:
    passage = "margin of conservatism in LGD estimation for defaulted exposures"
    question = "How much extra caution is needed when computing recovery rates?"
    assert lexical_overlap(question, passage) < 0.3


def test_lexical_overlap_empty_question() -> None:
    assert lexical_overlap("", "anything") == 0.0


def test_output_suffix_maps_filename_to_tag() -> None:
    assert output_suffix(Path("evaluation/ground_truth.csv")) == ""
    assert output_suffix(Path("x/ground_truth_hard.csv")) == "_hard"


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


# --- shingles / text coverage (the relevance primitive) --------------------- #
def test_shingles_are_order_sensitive() -> None:
    from evaluation.metrics import shingles

    # Same words, different order -> no shared 5-gram. This is why shingles beat
    # a bag of words in a corpus full of repeated regulatory phrasing.
    a = shingles("the institution shall apply a margin of conservatism")
    b = shingles("conservatism of margin a apply shall institution the")
    assert a and b and not (a & b)


def test_shingles_of_short_text_do_not_vanish() -> None:
    from evaluation.metrics import shingles

    assert shingles("only three words") == {("only", "three", "words")}
    assert shingles("") == set()


def test_text_coverage_is_one_for_a_containing_chunk() -> None:
    from evaluation.metrics import shingles, text_coverage

    passage = "institutions shall apply an appropriate margin of conservatism to estimates"
    wider = f"Some preceding sentence. {passage} And a following sentence."
    assert text_coverage(shingles(passage), wider) == 1.0


def test_text_coverage_is_zero_for_unrelated_text() -> None:
    from evaluation.metrics import shingles, text_coverage

    a = "institutions shall apply an appropriate margin of conservatism to estimates"
    b = "competent authorities should assess the rating system before granting permission"
    assert text_coverage(shingles(a), b) == 0.0


def test_text_coverage_is_partial_for_a_partial_window() -> None:
    from evaluation.metrics import shingles, text_coverage

    passage = " ".join(f"word{i}" for i in range(40))
    half = " ".join(f"word{i}" for i in range(20))
    assert 0.2 < text_coverage(shingles(passage), half) < 0.8


def test_relevant_by_text_requires_the_same_document() -> None:
    from evaluation.metrics import relevant_by_text, shingles

    passage = "institutions shall apply an appropriate margin of conservatism to estimates"
    sh = shingles(passage)
    assert relevant_by_text("d", passage, "d", sh) is True
    assert relevant_by_text("other", passage, "d", sh) is False


def test_coverage_threshold_admits_no_unreachable_ground_truth() -> None:
    from evaluation.metrics import TEXT_COVERAGE_THRESHOLD

    # Calibrated on the corpus: above ~0.3, naive windows can no longer cover
    # the largest ground-truth chunks and those rows become guaranteed misses.
    assert 0.0 < TEXT_COVERAGE_THRESHOLD <= 0.30


# --- torch threading (T8) ---------------------------------------------------- #
def test_configure_torch_threads_uses_all_cores() -> None:
    """Reranking is serialised by `Retriever._rerank_lock`, so only one runs at a
    time — capping torch to 1 thread left every other core idle on the eval's
    slowest stage."""
    import os

    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from evaluation.corpus import configure_torch_threads

    configure_torch_threads()
    assert torch.get_num_threads() == (os.cpu_count() or 1)
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


# --- default-selection guidance is ground-truth aware ------------------------ #
def test_only_the_debiased_run_recommends_setting_the_default() -> None:
    """The standard set flatters lexical retrieval, so its winner must never be
    presented as the config to ship."""
    from evaluation.eval_retrieval import _default_guidance, _gt_label

    standard = _default_guidance("")
    debiased = _default_guidance("_hard")
    assert "do NOT set this as the default" in standard
    assert "eval-retrieval-hard" in standard
    assert "set it as the default" in debiased
    assert "DE-BIASED" in _gt_label("_hard") and "STANDARD" in _gt_label("")
