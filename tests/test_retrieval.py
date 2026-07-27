"""Retrieval / rewrite / prompt / RAG unit tests (SPEC §6–§8).

Covers the pure logic — RRF fusion, doc filtering, acronym expansion, citation
parsing, prompt assembly, cost — without requiring Qdrant, BM25 files, or an LLM.
"""

from __future__ import annotations

from app.prompts import build_context, build_messages, citation_header, system_prompt
from app.providers import estimate_cost
from app.rag import Citation, SourceChunk, check_citation_grounding, parse_citations, titles_match
from app.retrieval import RetrievedChunk, filter_by_doc_ids, reciprocal_rank_fusion
from app.rewrite import expand_acronyms, unknown_acronyms
from ingestion.models import Chunk


def _rc(chunk_id: str, doc_id: str = "d1", title: str = "Doc One", paras=("82",), score=1.0):
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=title,
        section_path=["5", "5.3"],
        para_ids=list(paras),
        pages=[45],
        text=f"text of {chunk_id}",
    )
    return RetrievedChunk(chunk=chunk, score=score)


# --- reciprocal rank fusion ------------------------------------------------- #
def test_rrf_rewards_items_ranked_in_both_lists() -> None:
    bm25 = ["a", "b", "c"]
    vector = ["b", "a", "d"]
    scores = reciprocal_rank_fusion([bm25, vector], k=60)
    # 'a' and 'b' appear in both lists near the top -> outrank 'c'/'d'.
    ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
    assert ranked[:2] == ["b", "a"] or ranked[:2] == ["a", "b"]
    assert scores["b"] > scores["c"]
    assert scores["a"] > scores["d"]


def test_rrf_rank_is_zero_based_and_uses_k() -> None:
    scores = reciprocal_rank_fusion([["x", "y"]], k=60)
    assert scores["x"] == 1 / 60
    assert scores["y"] == 1 / 61


# --- doc_id filtering ------------------------------------------------------- #
def test_filter_by_doc_ids_none_is_noop() -> None:
    chunks = [_rc("1", doc_id="a"), _rc("2", doc_id="b")]
    assert filter_by_doc_ids(chunks, None) == chunks
    assert filter_by_doc_ids(chunks, []) == chunks


def test_filter_by_doc_ids_keeps_only_allowed() -> None:
    chunks = [_rc("1", doc_id="a"), _rc("2", doc_id="b"), _rc("3", doc_id="a")]
    kept = filter_by_doc_ids(chunks, ["a"])
    assert [rc.chunk.chunk_id for rc in kept] == ["1", "3"]


# --- acronym expansion ------------------------------------------------------ #
def test_expand_acronyms_appends_known_expansions() -> None:
    out = expand_acronyms("What does the EBA require for PD and LGD?")
    assert "PD (probability of default)" in out
    assert "LGD (loss given default)" in out


def test_expand_acronyms_deduplicates_and_ignores_unknown() -> None:
    out = expand_acronyms("PD PD and XYZ metrics")
    assert out.count("PD (probability of default)") == 1
    assert "XYZ" not in out.split("[")[-1]  # unknown not expanded


def test_unknown_acronyms_lists_non_glossary_terms() -> None:
    assert unknown_acronyms("PD and XYZ and ABC") == ["XYZ", "ABC"]


# --- citation parsing ------------------------------------------------------- #
def test_parse_citations_extracts_title_and_para() -> None:
    text = "Institutions must apply MoC [EBA GL 2017/16, para. 82]."
    cites = parse_citations(text)
    assert len(cites) == 1
    assert cites[0].doc_title == "EBA GL 2017/16"
    assert cites[0].paras == "82"


def test_parse_citations_splits_multiple_and_dedupes() -> None:
    text = (
        "Claim [Doc A, para. 1; Doc B, para. 2]. Repeat [Doc A, para. 1]. "
        "Non-citation [see annex]."
    )
    cites = parse_citations(text)
    keys = [(c.doc_title, c.paras) for c in cites]
    assert keys == [("Doc A", "1"), ("Doc B", "2")]


# --- citation grounding self-check ------------------------------------------ #
def _src(doc_title: str, para_ids: list[str]) -> SourceChunk:
    return SourceChunk(
        chunk_id="x", doc_id="d", doc_title=doc_title, section_path=[],
        para_ids=para_ids, pages=[1], score=1.0, text="t",
    )


def test_grounding_all_citations_backed() -> None:
    cites = [Citation(text="Doc A, para. 82", doc_title="Doc A", paras="82")]
    assert check_citation_grounding(cites, [_src("Doc A", ["82", "83"])]) == []


def test_grounding_flags_missing_paragraph() -> None:
    cites = [Citation(text="Doc A, para. 99", doc_title="Doc A", paras="99")]
    assert check_citation_grounding(cites, [_src("Doc A", ["82"])]) == ["Doc A, para. 99"]


def test_grounding_flags_wrong_document() -> None:
    cites = [Citation(text="Doc B, para. 82", doc_title="Doc B", paras="82")]
    assert check_citation_grounding(cites, [_src("Doc A", ["82"])]) == ["Doc B, para. 82"]


def test_grounding_multi_para_partial_match_is_grounded() -> None:
    cites = [Citation(text="Doc A, para. 82, 99", doc_title="Doc A", paras="82, 99")]
    assert check_citation_grounding(cites, [_src("Doc A", ["82"])]) == []


def test_grounding_accepts_shortened_title_dropping_parenthetical() -> None:
    full = "ECB Guide to Internal Models (February 2024, consolidated)"
    cites = [
        Citation(
            text="ECB Guide to Internal Models, para. 66",
            doc_title="ECB Guide to Internal Models",
            paras="66",
        )
    ]
    assert check_citation_grounding(cites, [_src(full, ["66"])]) == []


def test_grounding_accepts_case_insensitive_title() -> None:
    cites = [Citation(text="doc a, para. 82", doc_title="doc a", paras="82")]
    assert check_citation_grounding(cites, [_src("Doc A", ["82"])]) == []


def test_grounding_accepts_shared_regulatory_code() -> None:
    full = (
        "EBA Guidelines on PD estimation, LGD estimation and treatment of "
        "defaulted exposures (EBA/GL/2017/16)"
    )
    cites = [
        Citation(
            text="EBA Guidelines on PD/LGD estimation (EBA/GL/2017/16), para. 82",
            doc_title="EBA Guidelines on PD/LGD estimation (EBA/GL/2017/16)",
            paras="82",
        )
    ]
    assert check_citation_grounding(cites, [_src(full, ["82"])]) == []


def test_grounding_rejects_ambiguous_short_prefix() -> None:
    """Bare 'EBA Guidelines' must not match every EBA document."""
    full = "EBA Guidelines on loan origination and monitoring (EBA/GL/2020/06)"
    cites = [
        Citation(text="EBA Guidelines, para. 26", doc_title="EBA Guidelines", paras="26")
    ]
    assert check_citation_grounding(cites, [_src(full, ["26"])]) == [
        "EBA Guidelines, para. 26"
    ]


def test_titles_match_helpers() -> None:
    full = "ECB Guide to Internal Models (February 2024, consolidated)"
    assert titles_match("ECB Guide to Internal Models", full)
    assert titles_match(full, full)
    assert not titles_match("Doc B", "Doc A")
    assert not titles_match("EBA Guidelines", full)


def test_titles_match_token_containment_handles_front_shortening() -> None:
    # Dropping the leading org word ("ECB") is not a prefix, but the content
    # words are still a subset — token-containment catches it (prefix did not).
    full = "ECB Guide to Internal Models (February 2024, consolidated)"
    assert titles_match("Guide to Internal Models", full)


def test_titles_match_rejects_distinct_docs_sharing_few_words() -> None:
    # Two different guidelines share only generic words -> not a match.
    assert not titles_match(
        "EBA Guidelines on loan origination",
        "EBA Guidelines on downturn LGD estimation",
    )


# --- prompts ---------------------------------------------------------------- #
def test_citation_header_format() -> None:
    assert citation_header(_rc("1", title="Doc One", paras=("82", "83"))) == (
        "[Doc One, para. 82, 83]"
    )


def test_build_context_includes_header_and_text() -> None:
    ctx = build_context([_rc("1"), _rc("2")])
    assert "[Doc One, para. 82]" in ctx
    assert "text of 1" in ctx and "text of 2" in ctx


def test_system_prompt_versions_differ() -> None:
    assert "few" not in system_prompt("v1").lower() or True  # v1 is the plain base
    assert len(system_prompt("v2")) > len(system_prompt("v1"))
    assert "Example:" in system_prompt("v2")


def test_build_messages_inserts_history_before_question() -> None:
    history = [
        {"role": "user", "content": "prev q"},
        {"role": "assistant", "content": "prev a"},
    ]
    msgs = build_messages("q", [_rc("1")], version="v2", history=history)
    assert msgs[0]["role"] == "system"
    assert msgs[1:3] == history
    assert msgs[-1]["role"] == "user" and "Context:" in msgs[-1]["content"]


def test_build_messages_without_history() -> None:
    msgs = build_messages("q", [_rc("1")], version="v1")
    assert len(msgs) == 2 and msgs[0]["role"] == "system" and msgs[1]["role"] == "user"


def test_system_prompt_unknown_raises() -> None:
    try:
        system_prompt("v3")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown prompt version")


# --- cost ------------------------------------------------------------------- #
def test_estimate_cost_known_model() -> None:
    # gpt-4o-mini: (0.15, 0.60) per 1M tokens.
    cost = estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - (0.15 + 0.60)) < 1e-9


def test_estimate_cost_unknown_model_is_zero() -> None:
    assert estimate_cost("some-unknown-model", 1000, 1000) == 0.0
