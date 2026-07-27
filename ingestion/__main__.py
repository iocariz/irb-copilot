"""Ingestion pipeline entry point (SPEC §5).

    python -m ingestion                       # full pipeline for every document
    python -m ingestion --from-stage parse    # resume from an intermediate stage
    python -m ingestion --only-doc ebagl_2017_16
    python -m ingestion --chunker naive       # build the evaluation baseline

Each stage reads the previous stage's on-disk artifacts, so stages are
independently runnable and idempotent (SPEC §17).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings

from .chunk import chunk_document
from .download import Source, download_all, load_sources
from .index import index_chunks
from .models import Chunk, ParsedDoc
from .parse import parse_pdf

STAGES = ("download", "parse", "chunk", "index")


def main() -> None:
    args = _parse_args()
    data_dir = Path(settings.data_dir)
    raw_dir = Path(settings.raw_dir)
    parsed_dir = data_dir / "parsed"
    chunks_dir = data_dir / "chunks"

    sources = load_sources(data_dir / "sources.yaml")
    if args.only_doc:
        sources = [s for s in sources if s.id == args.only_doc]
        if not sources:
            raise SystemExit(f"--only-doc {args.only_doc!r} matched no document")

    start = STAGES.index(args.from_stage)
    active = STAGES[start:]

    if "download" in active:
        download_all(sources, raw_dir, only_doc=args.only_doc)
    if "parse" in active:
        _run_parse(sources, raw_dir, parsed_dir)
    if "chunk" in active:
        _run_chunk(sources, parsed_dir, chunks_dir, kind=args.chunker)
    if "index" in active:
        _run_index(sources, chunks_dir, kind=args.chunker, recreate=args.recreate)


def _run_parse(sources: list[Source], raw_dir: Path, parsed_dir: Path) -> None:
    parsed_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        pdf_path = raw_dir / f"{source.id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"[parse] missing {pdf_path}; run the download stage")
        doc = parse_pdf(pdf_path, doc_id=source.id, doc_title=source.title)
        out = parsed_dir / f"{source.id}.json"
        out.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        print(f"[parse] {source.id}: {len(doc.paragraphs)} paragraphs -> {out}")


def _run_chunk(
    sources: list[Source], parsed_dir: Path, chunks_dir: Path, *, kind: str
) -> None:
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


def _run_index(
    sources: list[Source], chunks_dir: Path, *, kind: str, recreate: bool
) -> None:
    chunks: list[Chunk] = []
    for source in sources:
        path = chunks_dir / f"{source.id}.{kind}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"[index] missing {path}; run the chunk stage")
        # Split on "\n" only — NOT str.splitlines(), which also breaks on
        # U+2028/U+2029/U+0085 that can appear unescaped inside chunk text.
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))
    index_chunks(chunks, recreate=recreate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingestion", description=__doc__)
    parser.add_argument("--from-stage", choices=STAGES, default="download")
    parser.add_argument("--only-doc", default=None)
    parser.add_argument("--chunker", choices=("structure", "naive"), default=settings.chunker)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="drop and recreate the Qdrant collection before indexing",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
