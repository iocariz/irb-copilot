"""Loaders for evaluation: on-disk chunk sets and parsed documents."""

from __future__ import annotations

from app.config import Settings, get_settings
from ingestion.models import Chunk, ParsedDoc


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
