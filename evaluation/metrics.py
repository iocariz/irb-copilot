"""Retrieval metrics and relevance rules (SPEC §11.2).

All pure functions so the metric maths is unit-testable without any index.

A per-query result is a list of booleans in rank order: ``relevances[i]`` is
whether the i-th retrieved chunk is relevant. Relevance itself depends on the
chunker: structure chunks match the ground-truth chunk id exactly; naive chunks
(whose ids differ) match on document + paragraph-anchor overlap.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

# Minimal stopword set so lexical-overlap reflects content words, not filler.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "be",
    "as", "by", "with", "that", "this", "these", "those", "it", "its", "at", "from",
    "which", "what", "when", "how", "may", "should", "shall", "must", "not", "no", "if",
})
_WORD_RE = re.compile(r"[a-z0-9]+")

# Word n-gram size for cross-chunker text matching. 5 is long enough that a
# match means shared running text rather than shared regulatory boilerplate.
_SHINGLE_N = 5
# A chunk counts as covering the ground-truth passage when it reproduces at
# least this fraction of it. Calibrated on the real corpus: 0.30 is the highest
# threshold at which no ground-truth row becomes unreachable for either chunker
# (a row no chunk can satisfy is a guaranteed miss that silently penalises that
# chunker). At 0.30 the rule admits a mean of 1.10 structure / 1.58 naive chunks
# per row, against 6.71 under the old paragraph-only rule. Naive windows are 500
# tokens while ground-truth chunks run to 1000, so a window can legitimately
# hold only part of a large chunk — hence a fractional, not exact, criterion.
TEXT_COVERAGE_THRESHOLD = 0.30


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def lexical_overlap(question: str, passage: str) -> float:
    """Fraction of the question's content words that appear in the passage.

    A proxy for lexical bias: LLM-generated questions that reuse the source's
    vocabulary score high (favouring lexical/BM25 retrieval); paraphrased,
    "de-biased" questions score lower. Range [0, 1].
    """
    q = _content_words(question)
    if not q:
        return 0.0
    return len(q & _content_words(passage)) / len(q)


def hit_rate_at_k(relevances: Sequence[Sequence[bool]], k: int) -> float:
    """Fraction of queries with at least one relevant chunk in the top-k."""
    if not relevances:
        return 0.0
    hits = sum(1 for rel in relevances if any(rel[:k]))
    return hits / len(relevances)


def mrr_at_k(relevances: Sequence[Sequence[bool]], k: int) -> float:
    """Mean reciprocal rank of the first relevant chunk within the top-k."""
    if not relevances:
        return 0.0
    total = 0.0
    for rel in relevances:
        for rank, is_rel in enumerate(rel[:k], start=1):
            if is_rel:
                total += 1.0 / rank
                break
    return total / len(relevances)


def relevant_by_chunk_id(retrieved_chunk_id: str, gt_chunk_id: str) -> bool:
    """Exact-id match, used for the structure chunker (its ids are the truth)."""
    return retrieved_chunk_id == gt_chunk_id


def relevant_by_paragraph(
    retrieved_doc_id: str,
    retrieved_para_ids: Sequence[str],
    gt_doc_id: str,
    gt_para_ids: Sequence[str],
) -> bool:
    """Same-document paragraph-anchor overlap.

    UNSAFE ON ITS OWN — kept only as the last-resort fallback for a ground-truth
    row whose chunk is no longer in the corpus. Regulatory documents restart
    paragraph numbering in every section (see `ingestion.models.chunk_id_for`),
    so paragraph "1" recurs dozens of times per document and this rule accepts
    every one of them. On the real corpus it admitted a mean of 6.7 chunks per
    ground-truth row (max 51), inflating hit-rate. Always pair it with a section
    or text check — see `relevant_by_section` and `relevant_by_text`.
    """
    if retrieved_doc_id != gt_doc_id:
        return False
    return bool(set(retrieved_para_ids) & set(gt_para_ids))


def shingles(text: str, n: int = _SHINGLE_N) -> set[tuple[str, ...]]:
    """Word n-grams of `text`, lowercased (pure).

    Order-sensitive, so unlike a bag of words it does not match two passages
    that merely share vocabulary — which matters in a corpus where boilerplate
    regulatory phrasing repeats constantly.
    """
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def text_coverage(gt_shingles: set[tuple[str, ...]], candidate_text: str) -> float:
    """Fraction of the ground-truth chunk's shingles present in `candidate_text`.

    Robust to the whitespace and token-boundary differences between chunkers (a
    naive window is a decoded token slice, so its junctions differ from the
    structure chunk's), while still requiring genuinely shared running text.
    """
    if not gt_shingles:
        return 0.0
    return len(gt_shingles & shingles(candidate_text)) / len(gt_shingles)


def relevant_by_text(
    retrieved_doc_id: str,
    candidate_text: str,
    gt_doc_id: str,
    gt_shingles: set[tuple[str, ...]],
    threshold: float = TEXT_COVERAGE_THRESHOLD,
) -> bool:
    """Same document AND the candidate reproduces enough of the ground-truth text.

    The relevance criterion for BOTH chunkers. The question was written from one
    specific passage, so the only sound test is whether the retrieved chunk
    actually contains that passage — not whether it happens to share a paragraph
    number. Being identical for structure and naive is also what makes the
    two-chunker comparison fair: neither is judged more leniently.

    It is immune to the two failure modes that broke the paragraph rule:
    paragraph numbers recurring across sections, and parse failures where one
    "paragraph" swallows a huge span (`ebagl_2017_16` para. 221 spans 51 chunks).
    """
    if retrieved_doc_id != gt_doc_id:
        return False
    return text_coverage(gt_shingles, candidate_text) >= threshold
