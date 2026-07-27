"""Indexing stage (SPEC §5, step 4).

Embeds chunks and upserts them into the Qdrant collection `irb_chunks` with
deterministic ids (so re-runs upsert, not duplicate), and builds a persistent
BM25 index over the same chunk texts for lexical retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.providers import embed_texts

from .models import Chunk

_UPSERT_BATCH = 128


def index_chunks(chunks: list[Chunk], *, recreate: bool = False) -> None:
    """Build both the vector index (Qdrant) and the lexical index (BM25)."""
    if not chunks:
        raise ValueError("index_chunks called with no chunks")
    build_qdrant(chunks, recreate=recreate)
    build_bm25(chunks)


def build_qdrant(chunks: list[Chunk], *, recreate: bool = False) -> None:
    """Embed chunk texts and upsert them into the Qdrant collection."""
    client = QdrantClient(url=settings.qdrant_url)
    _ensure_collection(client, recreate=recreate)

    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        vectors = embed_texts([c.text for c in batch])
        points = [
            PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.model_dump())
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=settings.qdrant_collection, points=points)
    print(f"[index] upserted {len(chunks)} chunks into '{settings.qdrant_collection}'")

    # Guardrail: deterministic ids mean re-runs upsert in place, but chunks that
    # disappeared or changed id in a re-ingest leave orphaned points behind. If
    # the collection is larger than what we just indexed, say so loudly.
    total = client.count(settings.qdrant_collection, exact=True).count
    warning = orphan_warning(total, len(chunks))
    if warning:
        print(warning)


def orphan_warning(collection_count: int, indexed_count: int) -> str | None:
    """Return a warning if the collection holds more points than were indexed."""
    if collection_count <= indexed_count:
        return None
    orphans = collection_count - indexed_count
    return (
        f"[index] WARNING: collection holds {collection_count} points but only "
        f"{indexed_count} were just indexed — {orphans} orphaned from earlier "
        f"runs. Re-run with --recreate for a clean rebuild."
    )


def _ensure_collection(client: QdrantClient, *, recreate: bool) -> None:
    exists = client.collection_exists(settings.qdrant_collection)
    if exists and recreate:
        client.delete_collection(settings.qdrant_collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


def build_bm25(chunks: list[Chunk]) -> None:
    """Build and persist a BM25 index plus a parallel chunk manifest to disk."""
    import bm25s

    out_dir = Path(settings.bm25_index_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = [c.text for c in chunks]
    tokens = bm25s.tokenize(corpus, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)
    retriever.save(str(out_dir))

    manifest = out_dir / "chunks.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
    print(f"[index] persisted BM25 index + manifest ({len(chunks)} chunks) to {out_dir}")
