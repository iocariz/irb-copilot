"""Indexing stage (SPEC §5, step 4).

Embeds chunks and upserts them into a Qdrant collection with deterministic ids
(so re-runs upsert, not duplicate), and builds a persistent BM25 index over the
same chunk texts. The collection name and BM25 directory are parameterisable so
the evaluation (SPEC §11.2) can build a separate naive-chunk index alongside the
production structure index.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import Settings, get_settings, settings
from app.providers import embed_texts

from .models import Chunk

_UPSERT_BATCH = 128


def index_targets(kind: str, cfg: Settings | None = None) -> tuple[str, Path]:
    """Return the (Qdrant collection, BM25 dir) an index of `kind` writes to.

    The naive baseline is isolated to a separate namespace (`*_naive`) so
    building it — via ingestion `--chunker naive` or the evaluation — never
    overwrites the production (structure) index.
    """
    cfg = cfg or get_settings()
    if kind == "naive":
        return f"{cfg.qdrant_collection}_naive", cfg.data_path / "bm25_index_naive"
    return cfg.qdrant_collection, cfg.bm25_index_dir


def index_chunks(
    chunks: list[Chunk],
    *,
    recreate: bool = False,
    collection: str | None = None,
    bm25_dir: Path | None = None,
) -> None:
    """Build both the vector index (Qdrant) and the lexical index (BM25)."""
    if not chunks:
        raise ValueError("index_chunks called with no chunks")
    collection = collection or settings.qdrant_collection
    bm25_dir = bm25_dir or settings.bm25_index_dir
    build_qdrant(chunks, recreate=recreate, collection=collection)
    build_bm25(chunks, bm25_dir=bm25_dir)


def build_qdrant(
    chunks: list[Chunk], *, recreate: bool = False, collection: str | None = None
) -> None:
    """Embed chunk texts and upsert them into the Qdrant collection."""
    collection = collection or settings.qdrant_collection
    client = QdrantClient(url=settings.qdrant_url)

    # Embed EVERYTHING first. `recreate` drops the collection, so if embedding
    # fails mid-way we must not have dropped it yet — otherwise a transient
    # OpenAI error would destroy the existing index with nothing to put back.
    points: list[PointStruct] = []
    for start in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[start : start + _UPSERT_BATCH]
        vectors = embed_texts([c.text for c in batch])
        points.extend(
            PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.model_dump())
            for chunk, vector in zip(batch, vectors, strict=True)
        )

    # Only now — with all vectors in hand — (re)create and upsert.
    _ensure_collection(client, collection, recreate=recreate)
    for start in range(0, len(points), _UPSERT_BATCH):
        client.upsert(collection_name=collection, points=points[start : start + _UPSERT_BATCH])
    print(f"[index] upserted {len(chunks)} chunks into '{collection}'")

    # Guardrail: deterministic ids mean re-runs upsert in place, but chunks that
    # disappeared or changed id in a re-ingest leave orphaned points behind. If
    # the collection is larger than what we just indexed, say so loudly.
    total = client.count(collection, exact=True).count
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


def _ensure_collection(client: QdrantClient, collection: str, *, recreate: bool) -> None:
    exists = client.collection_exists(collection)
    if exists and recreate:
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim, distance=Distance.COSINE
            ),
        )


def build_bm25(chunks: list[Chunk], *, bm25_dir: Path | None = None) -> None:
    """Build and persist a BM25 index plus a parallel chunk manifest to disk.

    Written to a temp sibling directory and swapped in atomically, so a failed or
    partial write never leaves a corrupt/half-written index at the live path (which
    would silently diverge from the Qdrant vectors that hybrid retrieval fuses).
    """
    import bm25s

    out_dir = bm25_dir or settings.bm25_index_dir
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir.parent / f".{out_dir.name}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        corpus = [c.text for c in chunks]
        tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokens)
        retriever.save(str(tmp_dir))

        manifest = tmp_dir / "chunks.jsonl"
        with manifest.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.model_dump(), ensure_ascii=False) + "\n")
        # Completeness check before the swap: the manifest must hold every chunk.
        written = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
        if written != len(chunks):
            raise OSError(f"BM25 manifest wrote {written} of {len(chunks)} chunks")

        # Atomic swap-in (same filesystem — tmp is a sibling of out_dir).
        if out_dir.exists():
            shutil.rmtree(out_dir)
        os.replace(tmp_dir, out_dir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[index] persisted BM25 index + manifest ({len(chunks)} chunks) to {out_dir}")
