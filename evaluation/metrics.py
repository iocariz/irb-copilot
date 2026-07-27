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
    """Same-document paragraph-anchor overlap, used for the naive chunker.

    Naive windows have different ids than the ground-truth (structure) chunk, so
    a hit is: same document AND the window covers at least one of the ground
    truth's paragraph numbers.
    """
    if retrieved_doc_id != gt_doc_id:
        return False
    return bool(set(retrieved_para_ids) & set(gt_para_ids))
