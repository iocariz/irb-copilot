"""UI rendering tests (SPEC §10).

Streamlit itself is not exercised; these cover the pure formatting helpers and
the feedback guard, which are where the user-visible defects were.
"""

from __future__ import annotations

import pytest

from app.ui import _feedback_state_key, format_pages, source_header


def _src(**overrides) -> dict:
    base = {
        "doc_title": "EBA Guidelines on PD/LGD estimation (EBA/GL/2017/16)",
        "para_ids": ["82"],
        "section_path": ["5 Margin of conservatism", "5.3 Estimation error"],
        "pages": [45],
    }
    return {**base, **overrides}


# --- page formatting --------------------------------------------------------- #
def test_single_page() -> None:
    assert format_pages([45]) == "p. 45"


def test_contiguous_pages_collapse_to_a_range() -> None:
    assert format_pages([12, 13, 14]) == "pp. 12–14"


def test_non_contiguous_pages_are_listed() -> None:
    assert format_pages([12, 14, 19]) == "pp. 12, 14, 19"


def test_no_pages_renders_nothing() -> None:
    assert format_pages([]) == ""


def test_pages_are_never_rendered_as_a_python_list() -> None:
    """The header interpolated the list directly: "(p. [12, 13])"."""
    for pages in ([1], [12, 13], [12, 14], []):
        assert "[" not in format_pages(pages) and "]" not in format_pages(pages)


# --- source headers ---------------------------------------------------------- #
def test_numbered_chunk_shows_its_paragraph_anchor() -> None:
    header = source_header(_src(para_ids=["82", "83"], pages=[45, 46]))
    assert "para. 82, 83" in header
    assert "pp. 45–46" in header


def test_unnumbered_chunk_falls_back_to_its_section() -> None:
    """24% of chunks carry no paragraph number by design — inventing one is what
    put 48% of the corpus under a wrong citation anchor. They used to render as a
    bare "— para." with nothing after it."""
    header = source_header(_src(para_ids=[], section_path=["Annex 1 Impact assessment"]))
    assert "para." not in header
    assert "§ Annex 1 Impact assessment" in header


def test_chunk_with_neither_paragraph_nor_section_is_still_labelled() -> None:
    header = source_header(_src(para_ids=[], section_path=[], pages=[]))
    assert "unnumbered" in header
    assert header.endswith("unnumbered")  # no trailing separator


@pytest.mark.parametrize(
    "src",
    [
        _src(),
        _src(para_ids=[]),
        _src(pages=[]),
        _src(para_ids=[], section_path=[], pages=[]),
    ],
)
def test_header_never_leaves_a_dangling_anchor(src) -> None:
    header = source_header(src)
    assert "para.  " not in header  # the empty-paragraph bug
    assert not header.rstrip().endswith("—")
    assert not header.rstrip().endswith("·")


# --- feedback guard ---------------------------------------------------------- #
def test_feedback_state_key_is_per_answer() -> None:
    """Keyed by answer id so rating one answer does not silence the buttons on
    the next."""
    assert _feedback_state_key("a") != _feedback_state_key("b")
    assert "a" in _feedback_state_key("a")


def test_section_label_that_repeats_the_document_title_is_skipped() -> None:
    """Would otherwise render "Basel III … — § Basel III …"."""
    header = source_header(
        _src(
            doc_title="Basel III: Finalising post-crisis reforms (BCBS d424)",
            para_ids=[],
            section_path=["Basel III: Finalising post-crisis reforms"],
            pages=[1],
        )
    )
    assert "§" not in header
    assert "unnumbered" in header


def test_a_meaningful_section_label_is_still_used() -> None:
    header = source_header(
        _src(para_ids=[], section_path=["4.2 Summary of responses"], pages=[70])
    )
    assert "§ 4.2 Summary of responses" in header


# --- retrieval score formatting (item 6) ------------------------------------ #
def test_score_is_always_labelled_with_its_mode() -> None:
    """A bare number reads as a confidence. It is not: cosine (0-1), BM25
    (unbounded), RRF (~1/60) and cross-encoder logits all appear here."""
    from app.ui import format_score

    assert format_score(0.87, "vector") == "vector score 0.87"
    assert format_score(7.4213, "hybrid_rerank") == "hybrid_rerank score 7.42"
    assert format_score(0.0164, "hybrid") == "hybrid score 0.0164"


def test_negative_cross_encoder_scores_render_sensibly() -> None:
    from app.ui import format_score

    assert format_score(-3.271, "hybrid_rerank") == "hybrid_rerank score -3.27"


def test_score_without_a_known_mode_still_renders() -> None:
    from app.ui import format_score

    assert format_score(1.5, None) == "score 1.5"


def test_scores_are_never_rendered_as_percentages_or_probabilities() -> None:
    from app.ui import format_score

    rendered = format_score(0.87, "vector")
    assert "%" not in rendered and "confidence" not in rendered.lower()
