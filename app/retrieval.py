"""Retrieval (SPEC §6).

Four modes selected by `RETRIEVAL_MODE`:

* ``bm25``          — lexical over chunk text (persisted bm25s index).
* ``vector``        — Qdrant cosine similarity over embeddings.
* ``hybrid``        — reciprocal rank fusion (RRF, k=60) of bm25 + vector.
* ``hybrid_rerank`` — hybrid top-20 → cross-encoder rerank → top-k.

All modes return the top-k chunks with scores and full metadata, and accept an
optional ``doc_ids`` metadata filter.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import ResponseHandlingException

from app.config import Settings, get_settings
from app.providers import embed_query
from ingestion.models import Chunk

# Candidate pool pulled from each source before fusion / reranking.
_FUSION_POOL = 60
_RERANK_POOL = 20
# Transport-level retries for Qdrant (see `Retriever._qdrant_search`).
_QDRANT_ATTEMPTS = 3
_QDRANT_BACKOFF = 0.25


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval, with its score."""

    chunk: Chunk
    score: float


def reciprocal_rank_fusion(
    rankings: list[list[str]], *, k: int = 60
) -> dict[str, float]:
    """Fuse ranked id lists into {id: score} via RRF: sum 1/(k + rank).

    Pure function (no I/O) so the fusion logic is unit-testable. Rank is 0-based
    (a common variant; the canonical Cormack et al. formula is 1-based). With the
    default k=60 the one-position offset is negligible, and the choice is internal
    and self-consistent — kept as-is deliberately so it doesn't change the ranking
    the evaluation was run against.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def filter_by_doc_ids(
    chunks: list[RetrievedChunk], doc_ids: list[str] | None
) -> list[RetrievedChunk]:
    """Keep only chunks whose doc_id is in `doc_ids` (no-op if None/empty)."""
    if not doc_ids:
        return chunks
    allowed = set(doc_ids)
    return [rc for rc in chunks if rc.chunk.doc_id in allowed]


class Retriever:
    """Loads the persisted BM25 index + manifest and a Qdrant client, and serves
    all four retrieval modes."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        collection: str | None = None,
        bm25_dir: Path | str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._collection = collection or self.settings.qdrant_collection
        self._bm25_dir = Path(bm25_dir) if bm25_dir else self.settings.bm25_index_dir
        # cached_property is NOT thread-safe (esp. Python 3.12, which dropped its
        # internal lock). Serialize the heavy first-loads so concurrent callers
        # can't build a torch/BM25/Qdrant resource twice at once (loading a torch
        # model from multiple threads simultaneously can crash the interpreter).
        self._load_lock = threading.Lock()
        # The cross-encoder's torch/tokenizers inference is not safe to call from
        # multiple threads at once (native double-free / crash). Callers run under
        # thread pools (the eval harness, FastAPI's sync-endpoint pool), so
        # serialize predict — it's ~100ms, while the parallelism that matters
        # (the LLM API calls) happens elsewhere.
        self._rerank_lock = threading.Lock()

    # --- lazily loaded resources ------------------------------------------- #
    @cached_property
    def _manifest(self) -> list[Chunk]:
        path = self._bm25_dir / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"BM25 manifest missing: {path}; run ingestion")
        # Split on "\n" only (chunk text may contain unicode line separators).
        return [
            Chunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]

    @cached_property
    def _bm25(self):  # noqa: ANN202 — bm25s type is optional at import time
        import bm25s

        with self._load_lock:
            if (cached := self.__dict__.get("_bm25")) is not None:
                return cached  # populated by another thread while we waited
            value = bm25s.BM25.load(str(self._bm25_dir), mmap=False)
            self.__dict__["_bm25"] = value  # publish inside the lock to close the race
            return value

    @cached_property
    def _qdrant(self) -> QdrantClient:
        with self._load_lock:
            if (cached := self.__dict__.get("_qdrant")) is not None:
                return cached
            value = QdrantClient(url=self.settings.qdrant_url)
            self.__dict__["_qdrant"] = value
            return value

    # --- public API -------------------------------------------------------- #
    def search(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        mode = mode or self.settings.retrieval_mode
        top_k = top_k or self.settings.top_k
        if mode == "bm25":
            return self._search_bm25(query, top_k, doc_ids)
        if mode == "vector":
            return self._search_vector(query, top_k, doc_ids)
        if mode == "hybrid":
            return self._search_hybrid(query, top_k, doc_ids)
        if mode == "hybrid_rerank":
            return self._search_hybrid_rerank(query, top_k, doc_ids)
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    # --- modes ------------------------------------------------------------- #
    def _search_bm25(
        self, query: str, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        import bm25s

        # Over-fetch when filtering so enough survive the doc_id filter.
        pool = top_k if not doc_ids else max(top_k * 10, 50)
        pool = min(pool, len(self._manifest))
        tokens = bm25s.tokenize(query, stopwords="en", show_progress=False)
        indices, scores = self._bm25.retrieve(
            tokens, k=pool, show_progress=False, return_as="tuple"
        )
        results = [
            RetrievedChunk(self._manifest[int(idx)], float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
        ]
        return filter_by_doc_ids(results, doc_ids)[:top_k]

    def _search_vector(
        self, query: str, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        pool = top_k if not doc_ids else max(top_k * 10, 50)
        vector = embed_query(query)
        hits = self._qdrant_search(vector, pool, doc_ids)
        return [
            RetrievedChunk(Chunk.model_validate(hit.payload), float(hit.score))
            for hit in hits
        ][:top_k]

    def _qdrant_search(self, vector: list[float], pool: int, doc_ids: list[str] | None):  # noqa: ANN202
        """Query Qdrant, retrying briefly on transport-level failures.

        One shared client serves a whole thread pool, and occasionally a pooled
        connection dies under load ("[Errno 9] Bad file descriptor"). That is a
        transport hiccup, not a bad request, and it is worth a retry: for a user
        request it avoids a spurious 502, and for the evaluation it avoids losing
        a config to a blip. `ResponseHandlingException` is raised only for
        transport errors — real API errors (missing collection, bad filter) are
        `UnexpectedResponse` and are not retried.
        """
        last: Exception | None = None
        for attempt in range(_QDRANT_ATTEMPTS):
            try:
                return self._qdrant.search(
                    collection_name=self._collection,
                    query_vector=vector,
                    limit=pool,
                    query_filter=_doc_filter(doc_ids),
                )
            except ResponseHandlingException as exc:
                last = exc
                if attempt + 1 < _QDRANT_ATTEMPTS:
                    time.sleep(_QDRANT_BACKOFF * (attempt + 1))
        raise RuntimeError(f"Qdrant search failed after {_QDRANT_ATTEMPTS} attempts") from last

    def _search_hybrid(
        self, query: str, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        bm = self._search_bm25(query, _FUSION_POOL, doc_ids)
        vec = self._search_vector(query, _FUSION_POOL, doc_ids)
        by_id = {rc.chunk.chunk_id: rc.chunk for rc in [*bm, *vec]}
        fused = reciprocal_rank_fusion(
            [[rc.chunk.chunk_id for rc in bm], [rc.chunk.chunk_id for rc in vec]],
            k=self.settings.rrf_k,
        )
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [RetrievedChunk(by_id[cid], score) for cid, score in ranked[:top_k]]

    def _search_hybrid_rerank(
        self, query: str, top_k: int, doc_ids: list[str] | None
    ) -> list[RetrievedChunk]:
        # Rerank at least top_k candidates; a larger top_k must not be silently
        # truncated by the fixed pool size.
        candidates = self._search_hybrid(query, max(top_k, _RERANK_POOL), doc_ids)
        if not candidates:
            return []
        pairs = [(query, rc.chunk.text) for rc in candidates]
        reranker = self._reranker
        with self._rerank_lock:  # native inference isn't safe to run concurrently
            scores = reranker.predict(pairs)
        reranked = sorted(
            zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True
        )
        return [RetrievedChunk(rc.chunk, float(score)) for rc, score in reranked[:top_k]]

    @cached_property
    def _reranker(self):  # noqa: ANN202 — sentence-transformers is optional
        from sentence_transformers import CrossEncoder

        with self._load_lock:
            if (cached := self.__dict__.get("_reranker")) is not None:
                return cached
            value = CrossEncoder(self.settings.rerank_model)
            self.__dict__["_reranker"] = value
            return value

    def warm(self, *, full: bool = False) -> None:
        """Eagerly load resources in the calling thread, so the first request
        isn't slow and concurrent lazy loads don't race under a thread pool.

        `full=True` loads the resources for ALL modes — the evaluation exercises
        every retrieval mode on one retriever, so it must warm everything up front
        rather than let the first parallel batch trigger the loads.
        """
        mode = self.settings.retrieval_mode
        _ = self._manifest
        if full or mode in ("bm25", "hybrid", "hybrid_rerank"):
            _ = self._bm25
        if full or mode in ("vector", "hybrid", "hybrid_rerank"):
            _ = self._qdrant
        if full or mode == "hybrid_rerank":
            _ = self._reranker


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """Process-wide shared Retriever, so the BM25 index, Qdrant client and
    cross-encoder load once and are reused across requests (not per call)."""
    return Retriever()


def _doc_filter(doc_ids: list[str] | None) -> qmodels.Filter | None:
    """Build a Qdrant payload filter restricting to `doc_ids`."""
    if not doc_ids:
        return None
    return qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=doc_ids))]
    )
