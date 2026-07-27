"""Stratified sampling for ground-truth generation (SPEC §11.1).

Sample ~N chunks stratified by document, allocating the budget proportionally to
each document's chunk count so every document is represented. Deterministic given
a seed, so ground-truth generation is reproducible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence

from ingestion.models import Chunk


def stratified_sample(chunks: Sequence[Chunk], n: int, *, seed: int) -> list[Chunk]:
    """Return up to `n` chunks, stratified by `doc_id`, reproducible via `seed`."""
    if n <= 0 or not chunks:
        return []

    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append(chunk)

    total = len(chunks)
    rng = random.Random(seed)
    sampled: list[Chunk] = []
    for doc_id in sorted(by_doc):
        group = by_doc[doc_id]
        # Proportional allocation, at least 1 per document that has chunks.
        take = max(1, round(n * len(group) / total))
        take = min(take, len(group))
        sampled.extend(rng.sample(group, take))

    # Trim any rounding overflow deterministically.
    rng.shuffle(sampled)
    return sampled[:n]
