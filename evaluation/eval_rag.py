"""RAG evaluation (SPEC §11.3).

    python -m evaluation.eval_rag [--n 100] [--seed 42] [--judge-model gpt-4o]

On a sampled set of questions, generates answers for every (prompt v1|v2) x
(model in EVAL_MODELS) config, then an LLM judge labels each answer
RELEVANT / PARTLY_RELEVANT / NON_RELEVANT and checks citation support. Records
cost + latency per config; writes results/rag_eval.csv and a distribution plot.
By default it also seeds the monitoring DB with judged conversations so the
Grafana judge-relevance panel has data.

Needs a running Qdrant + OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import PROJECT_ROOT, Settings, get_settings
from app.rag import answer as rag_answer
from evaluation.corpus import load_chunks
from evaluation.generate_ground_truth import GROUND_TRUTH_CSV
from evaluation.judge import RELEVANCE_LABELS, judge_answer
from monitoring.db import log_conversation

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
PROMPT_VERSIONS = ["v1", "v2"]


def load_sample(n: int, seed: int) -> list[dict]:
    if not GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(f"{GROUND_TRUTH_CSV} not found — generate ground truth first")
    with GROUND_TRUTH_CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return random.Random(seed).sample(rows, min(n, len(rows)))


def evaluate_config(
    gt: list[dict],
    reference: dict[str, str],
    settings: Settings,
    *,
    model: str,
    prompt_version: str,
    judge_model: str,
    log_to_db: bool,
    workers: int = 8,
) -> dict:
    """Answer + judge every question for one (model, prompt) config; aggregate.
    Questions are processed concurrently (I/O-bound answer + judge calls)."""
    cfg_settings = settings.model_copy(
        update={"llm_model": model, "prompt_version": prompt_version}
    )

    def answer_and_judge(row: dict) -> tuple[str, int, float, float, float]:
        ans = rag_answer(row["question"], settings=cfg_settings)
        # Label sources with their citation headers so the judge can verify that
        # the answer's [doc, para. X] citations actually map to a given source.
        sources = "\n\n".join(
            f"[{c.doc_title}, para. {', '.join(c.para_ids)}]\n{c.text}"
            for c in ans.chunks_used
        )
        verdict = judge_answer(
            row["question"], reference.get(row["chunk_id"], ""), ans.text, sources,
            model=judge_model,
        )
        if log_to_db:
            _safe_log(ans, verdict.relevance)
        return (
            verdict.relevance, int(verdict.citations_supported),
            ans.cost_usd, verdict.cost_usd, float(ans.latency_ms),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(answer_and_judge, gt))

    labels: Counter[str] = Counter(o[0] for o in outcomes)
    supported = sum(o[1] for o in outcomes)
    answer_cost = sum(o[2] for o in outcomes)
    judge_cost = sum(o[3] for o in outcomes)
    latency = sum(o[4] for o in outcomes)
    n = len(gt)
    return {
        "model": model,
        "prompt_version": prompt_version,
        "n": n,
        "relevant": labels["RELEVANT"],
        "partly_relevant": labels["PARTLY_RELEVANT"],
        "non_relevant": labels["NON_RELEVANT"],
        "relevant_rate": round(labels["RELEVANT"] / n, 4),
        "citation_supported_rate": round(supported / n, 4),
        "avg_answer_cost_usd": round(answer_cost / n, 8),
        "avg_judge_cost_usd": round(judge_cost / n, 8),
        "avg_latency_ms": round(latency / n, 1),
    }


def _safe_log(answer, relevance: str) -> None:  # noqa: ANN001
    try:
        log_conversation(answer, judge_relevance=relevance)
    except Exception as exc:  # noqa: BLE001 — DB seeding is best-effort.
        print(f"[eval-rag] log_conversation failed: {exc}")


def run(
    gt: list[dict], settings: Settings, *, judge_model: str, log_to_db: bool, workers: int = 8
) -> list[dict]:
    reference = {c.chunk_id: c.text for c in load_chunks("structure", settings)}
    results: list[dict] = []
    for model in settings.eval_models_list:
        for prompt_version in PROMPT_VERSIONS:
            print(f"[eval-rag] config model={model} prompt={prompt_version} …")
            row = evaluate_config(
                gt, reference, settings,
                model=model, prompt_version=prompt_version,
                judge_model=judge_model, log_to_db=log_to_db, workers=workers,
            )
            print(
                f"[eval-rag]   relevant={row['relevant_rate']:.2f} "
                f"cite_ok={row['citation_supported_rate']:.2f} "
                f"cost=${row['avg_answer_cost_usd']:.5f} lat={row['avg_latency_ms']}ms"
            )
            results.append(row)
    return results


def write_outputs(results: list[dict]) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "rag_eval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    _distribution_plot(results, RESULTS_DIR / "rag_eval.png")
    best = max(results, key=lambda r: (r["relevant_rate"], -r["avg_answer_cost_usd"]))
    print(f"\n[eval-rag] wrote {csv_path} and rag_eval.png")
    print(
        f"[eval-rag] BEST: model={best['model']} prompt={best['prompt_version']} "
        f"(relevant={best['relevant_rate']}, cite_ok={best['citation_supported_rate']})"
    )
    return best


def _distribution_plot(results: list[dict], out_path) -> None:
    labels = [f"{r['model']}|{r['prompt_version']}" for r in results]
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = [0.0] * len(results)
    colors = {"RELEVANT": "#55A868", "PARTLY_RELEVANT": "#DD8452", "NON_RELEVANT": "#C44E52"}
    keys = {
        "RELEVANT": "relevant",
        "PARTLY_RELEVANT": "partly_relevant",
        "NON_RELEVANT": "non_relevant",
    }
    for label in RELEVANCE_LABELS:
        values = [r[keys[label]] for r in results]
        ax.bar(x, values, bottom=bottom, label=label, color=colors[label])
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("answers")
    ax.set_title("RAG evaluation — judge relevance distribution by config")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    gt = load_sample(args.n, args.seed)
    n_configs = len(settings.eval_models_list) * len(PROMPT_VERSIONS)
    print(f"[eval-rag] {len(gt)} questions x {n_configs} configs (judge={args.judge_model})")
    results = run(
        gt, settings, judge_model=args.judge_model,
        log_to_db=not args.no_log_db, workers=args.workers,
    )
    write_outputs(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval_rag", description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--no-log-db", action="store_true", help="don't seed monitoring DB")
    parser.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return parser.parse_args()


if __name__ == "__main__":
    main()
