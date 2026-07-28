"""Chunker tests (SPEC §16 phase 2 deliverable).

The structure-aware merge/split logic is tested with an injected word-based token
counter so assertions are deterministic and independent of tiktoken's vocab. The
naive baseline and id determinism are tested against the real tokenizer.
"""

from __future__ import annotations

from ingestion.chunk import _split_by_sentences, chunk_naive, chunk_structured
from ingestion.models import Paragraph, ParsedDoc
from ingestion.tokens import count_tokens

# Deterministic, transparent token counter: one token per whitespace word.
words = lambda text: len(text.split())  # noqa: E731


def _para(pid: str, text: str, section: list[str], page: int = 1) -> Paragraph:
    return Paragraph(para_id=pid, text=text, section_path=section, page=page)


def _doc(paragraphs: list[Paragraph]) -> ParsedDoc:
    return ParsedDoc(doc_id="ebagl_2017_16", doc_title="Test Doc", paragraphs=paragraphs)


def test_small_same_subsection_paragraphs_are_merged() -> None:
    sec = ["5", "5.3 Margin of conservatism"]
    doc = _doc([_para("81", "a b c", sec), _para("82", "d e f", sec)])
    chunks = chunk_structured(doc, merge_below=250, count=words)
    assert len(chunks) == 1
    assert chunks[0].para_ids == ["81", "82"]
    assert "a b c" in chunks[0].text and "d e f" in chunks[0].text


def test_paragraphs_in_different_subsections_are_not_merged() -> None:
    doc = _doc(
        [
            _para("81", "a b c", ["5", "5.3 X"]),
            _para("82", "d e f", ["5", "5.4 Y"]),
        ]
    )
    chunks = chunk_structured(doc, merge_below=250, count=words)
    assert [c.para_ids for c in chunks] == [["81"], ["82"]]
    assert [c.section_path[-1] for c in chunks] == ["5.3 X", "5.4 Y"]


def test_merge_stops_when_combined_reaches_threshold() -> None:
    sec = ["1", "1.1"]
    # Each paragraph is 30 words; threshold 50 => only one fits per chunk.
    p = " ".join(["w"] * 30)
    doc = _doc([_para("1", p, sec), _para("2", p, sec), _para("3", p, sec)])
    chunks = chunk_structured(doc, merge_below=50, count=words)
    assert [c.para_ids for c in chunks] == [["1"], ["2"], ["3"]]


def test_oversized_chunk_is_hard_split_at_sentence_boundaries() -> None:
    sec = ["2", "2.1"]
    text = ". ".join(f"sentence number {i} here" for i in range(60)) + "."
    doc = _doc([_para("99", text, sec)])
    chunks = chunk_structured(doc, merge_below=250, hard_split=100, count=words)
    assert len(chunks) > 1
    assert all(words(c.text) <= 100 for c in chunks)
    # All sub-chunks keep the source paragraph's citation metadata.
    assert all(c.para_ids == ["99"] for c in chunks)
    # Hard-split sub-chunks get distinct ids.
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_oversized_single_sentence_is_token_split_below_cap() -> None:
    # A single sentence (no sentence boundaries) larger than the cap must still be
    # broken up — SPEC requires no chunk over the hard-split limit. Uses the real
    # tokenizer so the guard and the token-level split measure tokens the same way.
    sec = ["3", "3.1"]
    long_sentence = " ".join(["clause"] * 4000)  # no '. ' => one sentence
    doc = _doc([_para("100", long_sentence, sec)])
    chunks = chunk_structured(doc, merge_below=250, hard_split=1000, count=count_tokens)
    assert len(chunks) > 1
    assert all(count_tokens(c.text) <= 1000 for c in chunks)
    assert all(c.para_ids == ["100"] for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_split_by_sentences_hard_splits_leading_oversized_sentence() -> None:
    # Reproduces the reported bug: current is empty and the first sentence already
    # exceeds the cap, so the greedy packer previously emitted it whole.
    oversized = " ".join(["w"] * 3000)  # single sentence, no boundaries
    pieces = _split_by_sentences(oversized, hard_split=1000, count=count_tokens)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 1000 for p in pieces)


def test_pages_are_unioned_and_sorted() -> None:
    sec = ["3", "3.1"]
    doc = _doc([_para("10", "a b", sec, page=7), _para("11", "c d", sec, page=6)])
    chunks = chunk_structured(doc, merge_below=250, count=words)
    assert chunks[0].pages == [6, 7]


def test_repeated_paragraph_numbers_get_unique_ids() -> None:
    """Regression: regulatory docs restart numbering per section, so paragraph
    "1" recurs many times. Ordinal-based ids must still be unique within a doc,
    otherwise Qdrant upserts silently overwrite colliding chunks."""
    doc = _doc(
        [
            _para("1", "alpha", ["Article 1"]),
            _para("1", "beta", ["Article 2"]),
            _para("1", "gamma", []),
        ]
    )
    # merge_below=1 disables merging so each paragraph is its own chunk.
    chunks = chunk_structured(doc, merge_below=1, count=words)
    ids = [c.chunk_id for c in chunks]
    assert len(chunks) == 3
    assert len(set(ids)) == 3


def test_chunk_ids_are_deterministic_across_runs() -> None:
    sec = ["4", "4.1"]
    doc = _doc([_para("20", "hello world", sec)])
    first = chunk_structured(doc, count=words)
    second = chunk_structured(doc, count=words)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_naive_chunker_produces_overlapping_windows() -> None:
    sec = ["1", "1.1"]
    long_text = " ".join(f"token{i}" for i in range(1200))
    doc = _doc([_para("1", long_text, sec)])
    chunks = chunk_naive(doc, size=500, overlap=50)
    assert len(chunks) >= 2
    # Naive chunker ignores structure metadata by design.
    assert all(c.section_path == [] for c in chunks)
    assert all(c.chunk_id for c in chunks)


def test_stray_separators_are_normalized_and_jsonl_safe() -> None:
    """Regression: docling text can contain U+2028/U+2029/U+0085, which pydantic
    writes unescaped. Chunk text must strip them so a chunk round-trips as one
    JSONL line (str.splitlines would otherwise shatter it)."""
    dirty = "first part\u2028second part\u2029third part\x85end"
    doc = _doc([_para("1", dirty, ["1", "1.1"])])
    chunk = chunk_structured(doc, count=words)[0]
    assert not any(sep in chunk.text for sep in ("\u2028", "\u2029", "\x85"))
    # One serialized chunk must be exactly one physical line under splitlines().
    assert len(chunk.model_dump_json().splitlines()) == 1


def test_naive_empty_document_returns_no_chunks() -> None:
    # A document whose paragraphs are all empty tokenizes to nothing; return [] instead
    # of falling through to the window loop (which downstream would crash on).
    doc = _doc([_para("1", "", ["s"]), _para("2", "", ["s"])])
    assert chunk_naive(doc) == []


def test_naive_rejects_overlap_ge_size() -> None:
    doc = _doc([_para("1", "a b c", ["1"])])
    try:
        chunk_naive(doc, size=10, overlap=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")
