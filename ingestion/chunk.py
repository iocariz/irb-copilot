"""Chunking strategies (SPEC §5, step 3).

Two chunkers, selected by config/CLI:

* `structure` — structure-aware: one chunk per numbered paragraph; merge
  consecutive paragraphs under the same subsection while the combined length
  stays below `merge_below` tokens; hard-split any chunk over `hard_split`
  tokens at sentence boundaries.
* `naive` — fixed-size baseline (500 tokens, 50 overlap) used for the
  retrieval evaluation comparison.

Both build "raw" chunk contents first, then assign deterministic ids by ordinal
position (see `chunk_id_for`) so ids stay unique even when paragraph numbers
recur across sections.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from .models import Chunk, Paragraph, ParsedDoc, chunk_id_for
from .tokens import count_tokens, decode, encode, normalize_text, split_sentences

TokenCounter = Callable[[str], int]

MERGE_BELOW_TOKENS = 250
HARD_SPLIT_TOKENS = 1000
NAIVE_SIZE_TOKENS = 500
NAIVE_OVERLAP_TOKENS = 50


class _Raw(NamedTuple):
    """Chunk content before an id is assigned."""

    section_path: list[str]
    para_ids: list[str]
    pages: list[int]
    text: str


def chunk_document(doc: ParsedDoc, kind: str = "structure") -> list[Chunk]:
    """Dispatch to the requested chunker."""
    if kind == "structure":
        return chunk_structured(doc)
    if kind == "naive":
        return chunk_naive(doc)
    raise ValueError(f"unknown chunker kind: {kind!r}")


def _assign_ids(doc: ParsedDoc, raws: list[_Raw]) -> list[Chunk]:
    """Turn ordered raw contents into Chunks with deterministic, unique ids."""
    return [
        Chunk(
            chunk_id=chunk_id_for(doc.doc_id, raw.para_ids, ordinal),
            doc_id=doc.doc_id,
            doc_title=doc.doc_title,
            section_path=raw.section_path,
            para_ids=raw.para_ids,
            pages=raw.pages,
            text=normalize_text(raw.text),
        )
        for ordinal, raw in enumerate(raws)
    ]


# --------------------------------------------------------------------------- #
# Structure-aware chunker
# --------------------------------------------------------------------------- #
def chunk_structured(
    doc: ParsedDoc,
    *,
    merge_below: int = MERGE_BELOW_TOKENS,
    hard_split: int = HARD_SPLIT_TOKENS,
    count: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Merge small consecutive same-subsection paragraphs, then hard-split
    anything oversized. Preserves paragraph order and citation metadata."""
    raws: list[_Raw] = []
    paras = doc.paragraphs
    i = 0
    while i < len(paras):
        group = [paras[i]]
        group_tokens = count(paras[i].text)
        j = i + 1
        while j < len(paras) and paras[j].section_path == paras[i].section_path:
            combined = group_tokens + count(paras[j].text)
            if combined < merge_below:
                group.append(paras[j])
                group_tokens = combined
                j += 1
            else:
                break
        raws.extend(_finalize_group(group, hard_split=hard_split, count=count))
        i = j
    return _assign_ids(doc, raws)


def _finalize_group(
    group: list[Paragraph],
    *,
    hard_split: int,
    count: TokenCounter,
) -> list[_Raw]:
    """Build one or more raw chunks from a merged paragraph group."""
    text = "\n\n".join(p.text for p in group)
    para_ids = [p.para_id for p in group]
    pages = sorted({p.page for p in group})
    section_path = group[0].section_path

    if count(text) <= hard_split:
        return [_Raw(section_path, para_ids, pages, text)]

    return [
        _Raw(section_path, para_ids, pages, piece)
        for piece in _split_by_sentences(text, hard_split, count)
    ]


def _split_by_sentences(text: str, hard_split: int, count: TokenCounter) -> list[str]:
    """Greedily pack sentences into pieces of at most `hard_split` tokens."""
    pieces: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        candidate = " ".join([*current, sentence])
        if current and count(candidate) > hard_split:
            pieces.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


# --------------------------------------------------------------------------- #
# Naive baseline chunker (fixed window, token-level overlap)
# --------------------------------------------------------------------------- #
def chunk_naive(
    doc: ParsedDoc,
    *,
    size: int = NAIVE_SIZE_TOKENS,
    overlap: int = NAIVE_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Fixed-size sliding window over the whole document (SPEC §5 baseline)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than window size")

    # Track each token's page and source paragraph id, so a naive window records
    # which paragraphs it covers — needed to match naive chunks to ground truth
    # in the retrieval evaluation (SPEC §11.2).
    tokens: list[tuple[int, int, str]] = []
    for para in doc.paragraphs:
        for tok in encode(para.text):
            tokens.append((tok, para.page, para.para_id))

    raws: list[_Raw] = []
    step = size - overlap
    for start in range(0, max(len(tokens), 1), step):
        window = tokens[start : start + size]
        if not window:
            break
        text = decode([tok for tok, _, _ in window])
        pages = sorted({page for _, page, _ in window})
        para_ids = list(dict.fromkeys(pid for _, _, pid in window))  # ordered-unique
        raws.append(_Raw([], para_ids, pages, text))
        if start + size >= len(tokens):
            break
    return _assign_ids(doc, raws)
