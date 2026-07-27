"""Retrieval evaluation (SPEC §11.2).

    python -m evaluation.eval_retrieval [--limit N] [--rebuild-naive]

For every config in {bm25, vector, hybrid, hybrid_rerank} x {structure, naive}
x {rewrite off, on}, computes hit-rate@5 and MRR@5 against the ground truth and
writes results/retrieval_eval.csv plus a bar-chart PNG. Prints the best config,
which should become the documented default in .env.example.

Needs a running Qdrant + OPENAI_API_KEY (query embeddings, and the LLM rewrite);
`hybrid_rerank` also uses the cross-encoder (CPU, slower).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import PROJECT_ROOT, Settings, get_settings
from app.retrieval import RetrievedChunk, Retriever
from app.rewrite import rewrite_query
from evaluation.corpus import load_parsed_docs
from evaluation.generate_ground_truth import GROUND_TRUTH_CSV
from evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    relevant_by_chunk_id,
    relevant_by_paragraph,
)
from ingestion.chunk import chunk_document
from ingestion.index import index_chunks

MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
CHUNKERS = ["structure", "naive"]
REWRITE = [False, True]
TOP_K = 5

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def load_ground_truth(path: Path) -> list[dict]:
    """Load a ground-truth CSV, parsing para_ids into a list."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m evaluation.generate_ground_truth`"
        )
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["para_ids_list"] = [p for p in row["para_ids"].split(",") if p]
            rows.append(row)
    return rows


def output_suffix(gt_path: Path) -> str:
    """'' for ground_truth.csv, '_hard' for ground_truth_hard.csv, etc."""
    stem = gt_path.stem
    return "" if stem == "ground_truth" else "_" + stem.removeprefix("ground_truth_")


def ensure_naive_index(settings: Settings, *, rebuild: bool) -> tuple[str, object]:
    """Build the parallel naive-chunk index if needed; return (collection, dir)."""
    collection = f"{settings.qdrant_collection}_naive"
    bm25_dir = settings.data_path / "bm25_index_naive"
    if rebuild or not (bm25_dir / "chunks.jsonl").exists():
        print("[eval] building naive index (parse -> naive chunks -> index)")
        naive_chunks = [
            chunk
            for doc in load_parsed_docs(settings)
            for chunk in chunk_document(doc, "naive")
        ]
        index_chunks(naive_chunks, recreate=True, collection=collection, bm25_dir=bm25_dir)
    return collection, bm25_dir


def precompute_queries(gt: list[dict], settings: Settings) -> dict[tuple[str, bool], str]:
    """Map (question, rewrite_on) -> query string; rewrites each question once."""
    rw_settings = settings.model_copy(update={"enable_rewrite": True})
    queries: dict[tuple[str, bool], str] = {}
    for question in sorted({row["question"] for row in gt}):
        queries[(question, False)] = question
        queries[(question, True)] = rewrite_query(question, rw_settings).rewritten
    return queries


def _is_relevant(chunker: str, hit: RetrievedChunk, row: dict) -> bool:
    if chunker == "structure":
        return relevant_by_chunk_id(hit.chunk.chunk_id, row["chunk_id"])
    return relevant_by_paragraph(
        hit.chunk.doc_id, hit.chunk.para_ids, row["doc_id"], row["para_ids_list"]
    )


def evaluate_config(
    retriever: Retriever,
    gt: list[dict],
    queries: dict[tuple[str, bool], str],
    *,
    mode: str,
    chunker: str,
    rewrite_on: bool,
) -> tuple[float, float]:
    relevances: list[list[bool]] = []
    for row in gt:
        query = queries[(row["question"], rewrite_on)]
        hits = retriever.search(query, mode=mode, top_k=TOP_K)
        relevances.append([_is_relevant(chunker, h, row) for h in hits])
    return hit_rate_at_k(relevances, TOP_K), mrr_at_k(relevances, TOP_K)


def run(gt: list[dict], settings: Settings, *, rebuild_naive: bool) -> list[dict]:
    collection, bm25_dir = ensure_naive_index(settings, rebuild=rebuild_naive)
    retrievers = {
        "structure": Retriever(settings),
        "naive": Retriever(settings, collection=collection, bm25_dir=bm25_dir),
    }
    queries = precompute_queries(gt, settings)

    results: list[dict] = []
    for chunker in CHUNKERS:
        for mode in MODES:
            for rewrite_on in REWRITE:
                hr, mrr = evaluate_config(
                    retrievers[chunker], gt, queries,
                    mode=mode, chunker=chunker, rewrite_on=rewrite_on,
                )
                label = f"{mode}|{chunker}|{'rw' if rewrite_on else 'raw'}"
                print(f"[eval] {label:32s} hit@5={hr:.3f} mrr@5={mrr:.3f}")
                results.append(
                    {
                        "retrieval_mode": mode,
                        "chunker": chunker,
                        "rewrite": rewrite_on,
                        "hit_rate_at_5": round(hr, 4),
                        "mrr_at_5": round(mrr, 4),
                        "n_queries": len(gt),
                    }
                )
    return results


def _config_label(r: dict) -> str:
    return f"{r['retrieval_mode']}|{r['chunker']}|{'rw' if r['rewrite'] else 'raw'}"


def write_outputs(results: list[dict], suffix: str = "") -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"retrieval_eval{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    labels = [_config_label(r) for r in results]
    png_path = RESULTS_DIR / f"retrieval_eval{suffix}.png"
    _bar_chart(labels, [r["hit_rate_at_5"] for r in results], png_path)

    best = max(results, key=lambda r: (r["hit_rate_at_5"], r["mrr_at_5"]))
    print(f"\n[eval] wrote {csv_path} and {png_path.name}")
    print(
        f"[eval] BEST: mode={best['retrieval_mode']} chunker={best['chunker']} "
        f"rewrite={best['rewrite']} (hit@5={best['hit_rate_at_5']}, mrr@5={best['mrr_at_5']})"
    )
    print("[eval] -> set this as the default in .env.example")
    return best


def _bar_chart(labels: list[str], values: list[float], out_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("hit-rate@5")
    ax.set_title("Retrieval evaluation — hit-rate@5 by config")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    gt_path = Path(args.ground_truth)
    gt = load_ground_truth(gt_path)
    if args.limit:
        gt = gt[: args.limit]
    print(f"[eval] {len(gt)} ground-truth questions x {len(MODES) * len(CHUNKERS) * 2} configs")
    results = run(gt, settings, rebuild_naive=args.rebuild_naive)
    write_outputs(results, output_suffix(gt_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval_retrieval", description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap ground-truth rows")
    parser.add_argument("--rebuild-naive", action="store_true", help="rebuild naive index")
    parser.add_argument(
        "--ground-truth", default=str(GROUND_TRUTH_CSV),
        help="ground-truth CSV (e.g. evaluation/ground_truth_hard.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
