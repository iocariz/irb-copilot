"""Reusable ingestion stages (SPEC §5).

The four stages are plain functions so both the CLI entry point (`__main__`) and
the Prefect flow (`flow.py`) drive the exact same logic. Each stage reads the
previous stage's on-disk artifacts, so they stay independently runnable and
idempotent (SPEC §17).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from app.config import Settings, get_settings

from .chunk import chunk_document
from .download import Source, download_all, load_sources
from .index import index_chunks, index_targets
from .models import Chunk, ParsedDoc
from .parse import parse_pdf

STAGES = ("download", "parse", "chunk", "index")


class Paths(NamedTuple):
    data: Path
    raw: Path
    parsed: Path
    chunks: Path


def resolve_paths(settings: Settings | None = None) -> Paths:
    settings = settings or get_settings()
    data = settings.data_path
    return Paths(data=data, raw=settings.raw_path, parsed=data / "parsed", chunks=data / "chunks")


def load_source_list(settings: Settings | None = None, only_doc: str | None = None) -> list[Source]:
    settings = settings or get_settings()
    sources = load_sources(settings.data_path / "sources.yaml")
    if only_doc:
        sources = [s for s in sources if s.id == only_doc]
        if not sources:
            raise SystemExit(f"--only-doc {only_doc!r} matched no document")
    return sources


def run_download(sources: list[Source], raw_dir: Path, only_doc: str | None = None) -> None:
    download_all(sources, raw_dir, only_doc=only_doc)


def run_parse(sources: list[Source], raw_dir: Path, parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        pdf_path = raw_dir / f"{source.id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"[parse] missing {pdf_path}; run the download stage")
        doc = parse_pdf(pdf_path, doc_id=source.id, doc_title=source.title)
        out = parsed_dir / f"{source.id}.json"
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        print(f"[parse] {source.id}: {len(doc.paragraphs)} paragraphs -> {out}")


def run_chunk(sources: list[Source], parsed_dir: Path, chunks_dir: Path, *, kind: str) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        parsed = parsed_dir / f"{source.id}.json"
        if not parsed.exists():
            raise FileNotFoundError(f"[chunk] missing {parsed}; run the parse stage")
        doc = ParsedDoc.model_validate_json(parsed.read_text(encoding="utf-8"))
        chunks = chunk_document(doc, kind=kind)
        out = chunks_dir / f"{source.id}.{kind}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")
        print(f"[chunk] {source.id}: {len(chunks)} {kind} chunks -> {out}")


def run_index(chunks_dir: Path, *, kind: str, recreate: bool) -> None:
    """Index the FULL on-disk corpus for `kind` (every ``*.<kind>.jsonl``).

    The BM25 index is monolithic — `build_bm25` overwrites the persisted index
    with whatever chunks it is given. Indexing a subset (e.g. under --only-doc)
    would therefore shrink BM25 to that one document while Qdrant keeps the rest,
    silently desyncing lexical and vector retrieval. So the index stage always
    reads every chunk file; --only-doc only scopes download/parse/chunk.
    """
    files = sorted(chunks_dir.glob(f"*.{kind}.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"[index] no {kind} chunk files in {chunks_dir}; run the chunk stage"
        )
    chunks: list[Chunk] = []
    for path in files:
        # Split on "\n" only — NOT str.splitlines(), which also breaks on
        # U+2028/U+2029/U+0085 that can appear unescaped inside chunk text.
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    # Naive routes to an isolated namespace so it never overwrites production.
    collection, bm25_dir = index_targets(kind)
    print(
        f"[index] {len(chunks)} {kind} chunks from {len(files)} doc(s) "
        f"-> collection={collection}, bm25={bm25_dir.name}"
    )
    index_chunks(chunks, recreate=recreate, collection=collection, bm25_dir=bm25_dir)


def active_stages(from_stage: str, to_stage: str = "index") -> tuple[str, ...]:
    """The stages to run, inclusive, between from_stage and to_stage."""
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if end < start:
        raise ValueError(f"to_stage {to_stage!r} precedes from_stage {from_stage!r}")
    return STAGES[start : end + 1]
