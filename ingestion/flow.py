"""Prefect orchestration of the ingestion pipeline (SPEC §5).

    python -m ingestion.flow                            # full pipeline
    python -m ingestion.flow --from-stage parse         # resume from a stage
    python -m ingestion.flow --from-stage chunk --to-stage chunk
    python -m ingestion.flow --only-doc ebagl_2017_16 --chunker naive

Wraps the four ingestion stages as Prefect tasks so runs get task-level retries,
structured logging, and observability (viewable in the Prefect UI). Executes
locally with the default task runner — no Prefect server required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prefect import flow, get_run_logger, task

from app.config import get_settings

from . import pipeline
from .download import Source


@task(retries=2, retry_delay_seconds=10)
def download_stage(sources: list[Source], raw_dir: Path, only_doc: str | None) -> None:
    pipeline.run_download(sources, raw_dir, only_doc=only_doc)


@task
def parse_stage(sources: list[Source], raw_dir: Path, parsed_dir: Path) -> None:
    pipeline.run_parse(sources, raw_dir, parsed_dir)


@task
def chunk_stage(sources: list[Source], parsed_dir: Path, chunks_dir: Path, kind: str) -> None:
    pipeline.run_chunk(sources, parsed_dir, chunks_dir, kind=kind)


@task
def index_stage(sources: list[Source], chunks_dir: Path, kind: str, recreate: bool) -> None:
    pipeline.run_index(sources, chunks_dir, kind=kind, recreate=recreate)


@flow(name="irb-ingestion")
def ingestion_flow(
    from_stage: str = "download",
    to_stage: str = "index",
    only_doc: str | None = None,
    chunker: str = "structure",
    recreate: bool = False,
) -> None:
    logger = get_run_logger()
    settings = get_settings()
    paths = pipeline.resolve_paths(settings)
    sources = pipeline.load_source_list(settings, only_doc)
    active = pipeline.active_stages(from_stage, to_stage)
    logger.info("stages=%s docs=%d chunker=%s", active, len(sources), chunker)

    if "download" in active:
        download_stage(sources, paths.raw, only_doc)
    if "parse" in active:
        parse_stage(sources, paths.raw, paths.parsed)
    if "chunk" in active:
        chunk_stage(sources, paths.parsed, paths.chunks, chunker)
    if "index" in active:
        index_stage(sources, paths.chunks, chunker, recreate)


def main() -> None:
    args = _parse_args()
    ingestion_flow(
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        only_doc=args.only_doc,
        chunker=args.chunker,
        recreate=args.recreate,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingestion.flow", description=__doc__)
    parser.add_argument("--from-stage", choices=pipeline.STAGES, default="download")
    parser.add_argument("--to-stage", choices=pipeline.STAGES, default="index")
    parser.add_argument("--only-doc", default=None)
    parser.add_argument("--chunker", choices=("structure", "naive"), default="structure")
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
