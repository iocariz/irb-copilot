"""Ingestion pipeline entry point (SPEC §5).

    python -m ingestion                       # full pipeline for every document
    python -m ingestion --from-stage parse    # resume from an intermediate stage
    python -m ingestion --only-doc ebagl_2017_16
    python -m ingestion --chunker naive       # build the evaluation baseline

For orchestrated runs (retries, observability) see the Prefect flow in
`ingestion/flow.py` (`python -m ingestion.flow`), which drives the same stages.
"""

from __future__ import annotations

import argparse

from app.config import settings

from . import pipeline


def main() -> None:
    args = _parse_args()
    paths = pipeline.resolve_paths(settings)
    sources = pipeline.load_source_list(settings, args.only_doc)
    active = pipeline.active_stages(args.from_stage)

    if "download" in active:
        pipeline.run_download(sources, paths.raw, only_doc=args.only_doc)
    if "parse" in active:
        pipeline.run_parse(sources, paths.raw, paths.parsed)
    if "chunk" in active:
        pipeline.run_chunk(sources, paths.parsed, paths.chunks, kind=args.chunker)
    if "index" in active:
        # Index the full corpus (not just --only-doc) so BM25 and Qdrant agree.
        pipeline.run_index(paths.chunks, kind=args.chunker, recreate=args.recreate)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="ingestion", description=__doc__)
    parser.add_argument("--from-stage", choices=pipeline.STAGES, default="download")
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
