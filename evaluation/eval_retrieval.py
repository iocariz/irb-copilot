"""Retrieval evaluation (SPEC §11.2).

    python -m evaluation.eval_retrieval [--limit N] [--rebuild-naive]

For every config in {bm25, vector, hybrid, hybrid_rerank} x {structure, naive}
x {off, glossary, llm} rewriting — 24 configs — computes hit-rate@5 and MRR@5
against the ground truth and writes results/retrieval_eval.csv plus a bar-chart
PNG.

Prints the winning config. Only the DE-BIASED run (`--ground-truth
evaluation/ground_truth_hard.csv`) should set production defaults — standard
questions reuse the passages' wording and flatter lexical retrieval.

Each rewrite arm is produced by the production `rewrite_query`, so what is
measured is what ships.

Needs a running Qdrant + OPENAI_API_KEY (query embeddings, and the LLM rewrite);
`hybrid_rerank` also uses the cross-encoder (CPU, slower).
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import PROJECT_ROOT, Settings, get_settings
from app.providers import warm_query_embeddings
from app.retrieval import RetrievedChunk, Retriever
from app.rewrite import rewrite_query
from evaluation.corpus import configure_torch_threads, load_chunks, load_parsed_docs
from evaluation.generate_ground_truth import GROUND_TRUTH_CSV, check_ground_truth_freshness
from evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    relevant_by_paragraph,
    relevant_by_text,
    shingles,
)
from ingestion.chunk import chunk_document
from ingestion.index import index_chunks, index_targets
from ingestion.models import Chunk

MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
CHUNKERS = ["structure", "naive"]
# All three rewrite levels are ablated separately: `glossary` is what the app
# shipped under ENABLE_REWRITE=false and had never been measured against `off`.
REWRITE_MODES = ["off", "glossary", "llm"]
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
    collection, bm25_dir = index_targets("naive", settings)
    if rebuild or not (bm25_dir / "chunks.jsonl").exists():
        print("[eval] building naive index (parse -> naive chunks -> index)")
        naive_chunks = [
            chunk
            for doc in load_parsed_docs(settings)
            for chunk in chunk_document(doc, "naive")
        ]
        index_chunks(naive_chunks, recreate=True, collection=collection, bm25_dir=bm25_dir)
    return collection, bm25_dir


def precompute_queries(
    gt: list[dict], settings: Settings, *, workers: int = 8
) -> dict[tuple[str, str], str]:
    """Map (question, rewrite_mode) -> the query that mode actually retrieves on.

    Every arm — including ``off`` — goes through the production `rewrite_query`
    under a settings copy for that mode, so the evaluation measures the shipped
    code path by construction. The previous version hand-built the "rewrite off"
    arm as the raw question, which silently diverged from production: with
    ENABLE_REWRITE=false the app still applied glossary expansion, so the
    default config was never the config that was measured.

    Only the `llm` arm makes API calls; those are run concurrently.
    """
    questions = sorted({row["question"] for row in gt})
    queries: dict[tuple[str, str], str] = {}
    for mode in REWRITE_MODES:
        mode_settings = settings.model_copy(update={"rewrite_mode": mode})
        if mode == "llm":  # I/O-bound: one API call per question
            with ThreadPoolExecutor(max_workers=workers) as pool:
                rewritten = list(
                    # bind mode_settings now — the lambda outlives this iteration
                    pool.map(
                        lambda q, s=mode_settings: rewrite_query(q, s).rewritten, questions
                    )
                )
        else:  # pure/deterministic: no need for a pool
            rewritten = [rewrite_query(q, mode_settings).rewritten for q in questions]
        queries.update(zip(((q, mode) for q in questions), rewritten, strict=True))
    return queries


class GroundTruthIndex:
    """Resolves a ground-truth row to the corpus chunk it was written from.

    The ground-truth CSV stores only ids and paragraph anchors, but a sound
    relevance rule needs the source chunk's *text*. It is looked up here, and the
    derived shingle sets are cached — the eval asks the same question tens of
    thousands of times across 16 configs.

    Read concurrently from the per-config worker pool. No lock: the cache is
    idempotent (a race just recomputes the same shingle set) and both the dict
    assignment and the `missing` set insert are atomic under the GIL.
    """

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self._by_id = {c.chunk_id: c for c in chunks}
        self._shingles: dict[str, set[tuple[str, ...]]] = {}
        self.missing: set[str] = set()

    def chunk_for(self, gt_chunk_id: str) -> Chunk | None:
        chunk = self._by_id.get(gt_chunk_id)
        if chunk is None:
            self.missing.add(gt_chunk_id)
        return chunk

    def shingles_for(self, chunk: Chunk) -> set[tuple[str, ...]]:
        cached = self._shingles.get(chunk.chunk_id)
        if cached is None:
            cached = shingles(chunk.text)
            self._shingles[chunk.chunk_id] = cached
        return cached

    def report_missing(self) -> None:
        if self.missing:
            print(
                f"[eval] WARNING: {len(self.missing)} ground-truth chunk id(s) are not "
                f"in the current corpus — those rows fall back to the loose "
                f"doc+paragraph rule, which over-counts. Regenerate the ground truth."
            )


def _is_relevant(hit: RetrievedChunk, row: dict, gt: GroundTruthIndex) -> bool:
    """Is this retrieved chunk the passage the question was written from?

    Exact chunk id, or the chunk reproduces enough of the ground-truth passage's
    text (`relevant_by_text`) — one criterion for both chunkers.

    The previous rule, document + paragraph-anchor overlap, accepted a *mean of
    6.71 chunks per ground-truth row* on this corpus (max 51), because paragraph
    numbering restarts in every section. That inflated hit-rate@5 and MRR@5.
    Anchoring on the text instead brings it to 1.10 (structure) / 1.58 (naive).

    Matching on section path + paragraph was tried first and is not enough: it
    leaves parse failures where a single "paragraph" spans 51 chunks in one
    section (mean 4.28, max 51). Only the text test pins a hit to the passage.
    """
    chunk = hit.chunk
    if chunk.chunk_id == row["chunk_id"]:
        return True
    gt_chunk = gt.chunk_for(row["chunk_id"])
    if gt_chunk is None:  # stale ground truth: no text to compare, warn and degrade
        return relevant_by_paragraph(
            chunk.doc_id, chunk.para_ids, row["doc_id"], row["para_ids_list"]
        )
    return relevant_by_text(
        chunk.doc_id, chunk.text, row["doc_id"], gt.shingles_for(gt_chunk)
    )


def evaluate_config(
    retriever: Retriever,
    gt: list[dict],
    queries: dict[tuple[str, str], str],
    gt_index: GroundTruthIndex,
    *,
    mode: str,
    rewrite_mode: str,
    workers: int = 8,
) -> tuple[float, float]:
    def relevances_for(row: dict) -> list[bool]:
        query = queries[(row["question"], rewrite_mode)]
        hits = retriever.search(query, mode=mode, top_k=TOP_K)
        return [_is_relevant(h, row, gt_index) for h in hits]

    # bm25 is local (fast, single-threaded); other modes hit the embedding API.
    pool_workers = 1 if mode == "bm25" else workers
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        relevances = list(pool.map(relevances_for, gt))
    return hit_rate_at_k(relevances, TOP_K), mrr_at_k(relevances, TOP_K)


def run(gt: list[dict], settings: Settings, *, rebuild_naive: bool, workers: int = 8) -> list[dict]:
    # The relevance rule needs each ground-truth chunk's section path and text,
    # which the CSV doesn't carry — resolve them from the structure corpus.
    gt_index = GroundTruthIndex(load_chunks("structure", settings))
    collection, bm25_dir = ensure_naive_index(settings, rebuild=rebuild_naive)
    retrievers = {
        "structure": Retriever(settings),
        "naive": Retriever(settings, collection=collection, bm25_dir=bm25_dir),
    }
    # Load every mode's resources up front (single-threaded); the per-config eval
    # runs searches in a thread pool and concurrent first-loads aren't thread-safe.
    for retriever in retrievers.values():
        retriever.warm(full=True)
    queries = precompute_queries(gt, settings, workers=workers)
    # Every vector/hybrid config re-embeds the same query strings. Embed each one
    # once, in batches of 128, before the configs run: ~14,400 single-item
    # requests collapse to ~19 batched ones.
    warmed = warm_query_embeddings(queries.values())
    print(f"[eval] pre-embedded {warmed} distinct queries (batched)")

    results: list[dict] = []
    for chunker in CHUNKERS:
        for mode in MODES:
            for rewrite_mode in REWRITE_MODES:
                # hybrid_rerank reranks on CPU; loads are thread-safe and torch is
                # capped to 1 intra-op thread (see main), so the worker pool gives
                # parallelism without oversubscribing — no longer forced single-threaded.
                label = f"{mode}|{chunker}|{rewrite_mode}"
                try:
                    hr, mrr = evaluate_config(
                        retrievers[chunker], gt, queries, gt_index,
                        mode=mode, rewrite_mode=rewrite_mode, workers=workers,
                    )
                except Exception as exc:  # noqa: BLE001 — don't discard completed configs
                    # A full run is 24 configs over hundreds of questions; losing
                    # all of it to one transient fault (a dropped Qdrant
                    # connection, a rate limit outlasting retries) is the worst
                    # outcome. Same policy as eval_rag.
                    print(
                        f"[eval] SKIPPED {label} after error: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                print(f"[eval] {label:36s} hit@5={hr:.3f} mrr@5={mrr:.3f}")
                results.append(
                    {
                        "retrieval_mode": mode,
                        "chunker": chunker,
                        "rewrite_mode": rewrite_mode,
                        "hit_rate_at_5": round(hr, 4),
                        "mrr_at_5": round(mrr, 4),
                        "n_queries": len(gt),
                    }
                )
    gt_index.report_missing()
    return results


def _config_label(r: dict) -> str:
    return f"{r['retrieval_mode']}|{r['chunker']}|{r['rewrite_mode']}"


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
        f"[eval] BEST on {_gt_label(suffix)}: mode={best['retrieval_mode']} "
        f"chunker={best['chunker']} rewrite_mode={best['rewrite_mode']} "
        f"(hit@5={best['hit_rate_at_5']}, mrr@5={best['mrr_at_5']})"
    )
    print(_default_guidance(suffix))
    return best


def _gt_label(suffix: str) -> str:
    return "the DE-BIASED ground truth" if suffix == "_hard" else "the STANDARD ground truth"


def _default_guidance(suffix: str) -> str:
    """What to do with the winner — which depends on which ground truth ran.

    Only the de-biased set should pick production defaults. Standard-set
    questions are LLM-generated from the passages and reuse their vocabulary
    (mean lexical overlap ~0.83), which flatters lexical retrieval: BM25 tends to
    win there for a reason that does not survive real user phrasing. Emitting a
    blanket "set this as the default" after either run invites shipping exactly
    the config this project's own analysis argues against.
    """
    if suffix == "_hard":
        return "[eval] -> this is the de-biased winner: set it as the default in .env.example"
    return (
        "[eval] -> do NOT set this as the default: these questions reuse the "
        "passages' vocabulary, which flatters lexical search. Pick defaults from "
        "`make eval-retrieval-hard`; this run is the contrast that shows the bias."
    )


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
    configure_torch_threads()  # rerank is lock-serialised, so give it every core
    settings = get_settings()
    gt_path = Path(args.ground_truth)
    gt = load_ground_truth(gt_path)
    staleness = check_ground_truth_freshness(gt_path, settings)
    if staleness:
        print(staleness)
    if args.limit:
        gt = gt[: args.limit]
    n_configs = len(MODES) * len(CHUNKERS) * len(REWRITE_MODES)
    print(f"[eval] {len(gt)} ground-truth questions x {n_configs} configs")
    results = run(gt, settings, rebuild_naive=args.rebuild_naive, workers=args.workers)
    if not results:
        raise SystemExit("[eval] no configs completed — see the errors above")
    if len(results) < n_configs:
        print(f"[eval] NOTE: {n_configs - len(results)}/{n_configs} configs were skipped")
    write_outputs(results, output_suffix(gt_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval_retrieval", description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="cap ground-truth rows")
    parser.add_argument("--rebuild-naive", action="store_true", help="rebuild naive index")
    parser.add_argument(
        "--ground-truth", default=str(GROUND_TRUTH_CSV),
        help="ground-truth CSV (e.g. evaluation/ground_truth_hard.csv)",
    )
    parser.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return parser.parse_args()


if __name__ == "__main__":
    main()
