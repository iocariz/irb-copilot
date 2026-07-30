"""Loaders for evaluation: on-disk chunk sets and parsed documents."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence

from app.config import Settings, get_settings
from ingestion.models import Chunk, ParsedDoc


def configure_torch_threads() -> None:
    """Let the CPU cross-encoder use every core.

    This used to pin torch to 1 thread. That made sense when reranking ran
    concurrently across the worker pool — cores-many OpenMP threads per worker
    oversubscribed the CPU and thrashed. `Retriever._rerank_lock` was added
    afterwards and serialises `predict()`, so only ever one rerank runs at a
    time: the cap then left every core but one idle, on the evaluation's slowest
    stage (~64k query-passage pairs). The lock rules out oversubscription, so
    give torch the whole machine.

    The Python workers that keep running alongside are waiting on HTTP, so they
    cost essentially no CPU. No-op if torch isn't installed.
    """
    # Stop HF tokenizers from spawning its own thread pool when we already run a
    # Python thread pool (source of the "leaked semaphore" warnings / deadlocks).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch

        torch.set_num_threads(os.cpu_count() or 1)
    except ImportError:
        pass


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Stable hash of a chunk set's ids — changes iff the corpus is re-chunked.

    Ground truth records chunk ids (and paragraph anchors derived from them); a
    re-ingest that changes chunk ids silently invalidates those references. Stamp
    this fingerprint alongside the ground truth to detect that drift.
    """
    joined = "\n".join(sorted(c.chunk_id for c in chunks))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def load_chunks(kind: str, settings: Settings | None = None) -> list[Chunk]:
    """Load all chunks of a given kind ('structure' | 'naive') from data/chunks."""
    settings = settings or get_settings()
    chunks_dir = settings.data_path / "chunks"
    files = sorted(chunks_dir.glob(f"*.{kind}.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no {kind} chunk files in {chunks_dir}; run ingestion first"
        )
    chunks: list[Chunk] = []
    for path in files:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    return chunks


def load_parsed_docs(settings: Settings | None = None) -> list[ParsedDoc]:
    """Load all parsed documents from data/parsed."""
    settings = settings or get_settings()
    parsed_dir = settings.data_path / "parsed"
    files = sorted(parsed_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no parsed docs in {parsed_dir}; run ingestion first")
    return [ParsedDoc.model_validate_json(p.read_text(encoding="utf-8")) for p in files]
