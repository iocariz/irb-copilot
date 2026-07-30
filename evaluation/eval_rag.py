"""RAG evaluation (SPEC §11.3).

    python -m evaluation.eval_rag [--n 100] [--seed 42] [--judge-model gpt-5.4-mini]

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
from app.providers import (
    QuotaExhausted,
    max_concurrency,
    raise_if_quota_exhausted,
    require_pricing,
)
from app.rag import answer as rag_answer
from app.retrieval import get_retriever
from evaluation.corpus import configure_torch_threads, load_chunks
from evaluation.generate_ground_truth import GROUND_TRUTH_CSV, check_ground_truth_freshness
from evaluation.judge import PARSE_ERROR, RELEVANCE_LABELS, judge_answer
from monitoring.db import log_conversation

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
PROMPT_VERSIONS = ["v1", "v2"]
# A question that produced no answer at all (retrieval or model failure), as
# distinct from PARSE_ERROR (answered, but the verdict was unreadable).
ANSWER_ERROR = "ANSWER_ERROR"


def model_family(model: str) -> str:
    """The version family of a model name ('gpt-4o-mini' -> 'gpt-4o').

    Used to detect self-preference bias. Exact-name equality is too narrow: a
    gpt-5.4 judge scoring gpt-5.4-mini answers is the same failure mode, so
    compare families, not names.
    """
    return "-".join(model.split("-")[:2])


def is_self_judged(answer_model: str, judge_model: str) -> bool:
    """True if judge and answer model come from the same family (biased)."""
    return model_family(answer_model) == model_family(judge_model)


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
        try:
            return _answer_and_judge_one(row)
        except Exception as exc:  # noqa: BLE001 — one question must not sink the config
            # Billing exhaustion is permanent: every later question would fail too,
            # leaving a biased subsample. Abort instead of degrading the result.
            raise_if_quota_exhausted(exc)
            # A sustained rate limit can outlast the SDK's retries. Losing this
            # question is bad; losing the other 99 with it is worse. Reported as
            # ANSWER_ERROR and excluded from the rates, never scored as a bad answer.
            print(
                f"[eval-rag]   question failed ({type(exc).__name__}): "
                f"{str(exc)[:120]} — excluded from this config's rates"
            )
            return (ANSWER_ERROR, 0, 0.0, 0.0, 0.0)

    def _answer_and_judge_one(row: dict) -> tuple[str, int, float, float, float]:
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
            # No readable verdict -> log NULL, not "PARSE_ERROR": the Grafana
            # relevance panel should show real judgements only.
            _safe_log(ans, verdict.relevance if verdict.parsed else None)
        return (
            verdict.relevance, int(verdict.citations_supported),
            ans.cost_usd, verdict.cost_usd, float(ans.latency_ms),
        )

    # Pace models whose TPM cannot sustain the requested concurrency (gpt-4o).
    effective = max_concurrency(model, workers)
    if effective < workers:
        print(
            f"[eval-rag]   pacing {model} to {effective} concurrent request(s) "
            f"(requested {workers}) — its TPM cannot sustain more"
        )
    with ThreadPoolExecutor(max_workers=effective) as pool:
        outcomes = list(pool.map(answer_and_judge, gt))

    labels: Counter[str] = Counter(o[0] for o in outcomes)
    answer_cost = sum(o[2] for o in outcomes)
    judge_cost = sum(o[3] for o in outcomes)
    # Only answered questions have a real latency; a failure contributes 0.0 and
    # would drag the percentiles down.
    latencies = [o[4] for o in outcomes if o[0] != ANSWER_ERROR]
    latency = sum(latencies)
    n = len(gt)
    # Questions that produced no answer at all (vs. answered but unjudgeable).
    answer_errors = labels[ANSWER_ERROR]
    answered = n - answer_errors
    # Quality rates are over answers the judge actually returned a verdict for.
    # An unreadable judgement is missing data, not a NON_RELEVANT answer; scoring
    # it as one would understate every config, and most the config whose answers
    # make the judge most verbose.
    judged_outcomes = [o for o in outcomes if o[0] in RELEVANCE_LABELS]
    judged = len(judged_outcomes)
    supported = sum(o[1] for o in judged_outcomes)
    return {
        "model": model,
        "prompt_version": prompt_version,
        "n": n,
        # n -> answered -> judged, each narrowing for a different reason, so a
        # shrunken denominator is always visible rather than implied.
        "answered": answered,
        "answer_errors": answer_errors,
        "judged": judged,
        "parse_errors": labels[PARSE_ERROR],
        # A judge scoring answers from its own model family is prone to
        # self-preference bias; record it so the CSV/plot consumer can discount it.
        "self_judged": is_self_judged(model, judge_model),
        "relevant": labels["RELEVANT"],
        "partly_relevant": labels["PARTLY_RELEVANT"],
        "non_relevant": labels["NON_RELEVANT"],
        "relevant_rate": _rate(labels["RELEVANT"], judged),
        "citation_supported_rate": _rate(supported, judged),
        # Cost averages over questions that produced an answer — it was really
        # spent, whether or not the judge could read the verdict.
        "avg_answer_cost_usd": round(answer_cost / answered, 8) if answered else 0.0,
        "avg_judge_cost_usd": round(judge_cost / answered, 8) if answered else 0.0,
        # Latency percentiles, NOT just the mean. A handful of requests stall for
        # minutes under concurrency (SDK backoff, or queueing behind the
        # lock-serialised cross-encoder), and a single 10-minute outlier moved a
        # config's mean from ~3s to ~18s — making a cheap model look 7x slower
        # than an expensive one. The median is the number that describes the
        # typical request; p95 shows the tail without letting it dominate.
        "p50_latency_ms": _percentile(latencies, 0.50),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_latency_ms": round(latency / answered, 1) if answered else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of `values` (0.0 when empty)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))
    return round(ordered[index], 1)


def _rate(count: int, denominator: int) -> float:
    """A rate over successfully judged answers (0.0 when nothing was judged)."""
    return round(count / denominator, 4) if denominator else 0.0


def _safe_log(answer, relevance: str) -> None:  # noqa: ANN001
    try:
        log_conversation(answer, judge_relevance=relevance)
    except Exception as exc:  # noqa: BLE001 — DB seeding is best-effort.
        print(f"[eval-rag] log_conversation failed: {exc}")


def run(
    gt: list[dict], settings: Settings, *, judge_model: str, log_to_db: bool, workers: int = 8
) -> list[dict]:
    reference = {c.chunk_id: c.text for c in load_chunks("structure", settings)}
    # Load index/model once in the main thread; the per-question answers run in a
    # thread pool and concurrent first-loads of the reranker are not thread-safe.
    get_retriever().warm()
    results: list[dict] = []
    for model in settings.eval_models_list:
        if is_self_judged(model, judge_model):
            print(
                f"[eval-rag] WARNING: judge ({judge_model}) shares a model family with "
                f"the answer model ({model}) — self-preference bias likely; use a "
                f"different-family judge (--judge-model) for a trustworthy relevance "
                f"score on this config."
            )
        for prompt_version in PROMPT_VERSIONS:
            print(f"[eval-rag] config model={model} prompt={prompt_version} …")
            try:
                row = evaluate_config(
                    gt, reference, settings,
                    model=model, prompt_version=prompt_version,
                    judge_model=judge_model, log_to_db=log_to_db, workers=workers,
                )
            except QuotaExhausted:
                # Not skippable: no later config can succeed either. Write what
                # completed, then stop.
                print("[eval-rag] ABORTING: account quota exhausted — top up billing")
                return results
            except Exception as exc:  # noqa: BLE001 — don't discard completed configs
                # e.g. a sustained rate limit that outlasts retries. Log loudly and
                # keep going so the configs that finished still get written.
                print(
                    f"[eval-rag]   SKIPPED model={model} prompt={prompt_version} "
                    f"after error: {type(exc).__name__}: {exc}"
                )
                continue
            print(
                f"[eval-rag]   relevant={row['relevant_rate']:.2f} "
                f"cite_ok={row['citation_supported_rate']:.2f} "
                f"cost=${row['avg_answer_cost_usd']:.5f} lat={row['avg_latency_ms']}ms "
                f"(judged {row['judged']}/{row['n']})"
            )
            if row["answer_errors"]:
                print(
                    f"[eval-rag]   WARNING: {row['answer_errors']}/{row['n']} questions "
                    f"produced no answer and are excluded from the rates above"
                )
            if row["parse_errors"]:
                # Rates are over `judged`, so a large count means a smaller and
                # possibly non-random sample — say so rather than let the rate
                # look like it was computed over everything.
                print(
                    f"[eval-rag]   WARNING: {row['parse_errors']}/{row['n']} judge "
                    f"verdicts were unreadable and are excluded from the rates above"
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
    # Only complete samples may win: a config that lost questions mid-run has a
    # non-random subsample, so its rate is not comparable.
    complete = [r for r in results if r["answered"] == r["n"]]
    if len(complete) < len(results):
        incomplete = [
            f"{r['model']}|{r['prompt_version']} ({r['answered']}/{r['n']})"
            for r in results if r["answered"] != r["n"]
        ]
        print(
            f"[eval-rag] NOTE: excluded from BEST (incomplete sample): "
            f"{', '.join(incomplete)}"
        )
    best = max(complete or results, key=lambda r: (r["relevant_rate"], -r["avg_answer_cost_usd"]))
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
    colors = {
        "RELEVANT": "#55A868",
        "PARTLY_RELEVANT": "#DD8452",
        "NON_RELEVANT": "#C44E52",
        # Grey, and stacked on top: missing data, not a verdict. Plotted so a
        # judge failure is visible in the artifact rather than shrinking the bar.
        PARSE_ERROR: "#B0B0B0",
    }
    keys = {
        "RELEVANT": "relevant",
        "PARTLY_RELEVANT": "partly_relevant",
        "NON_RELEVANT": "non_relevant",
        PARSE_ERROR: "parse_errors",
    }
    for label in (*RELEVANCE_LABELS, PARSE_ERROR):
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
    configure_torch_threads()  # rerank is lock-serialised, so give it every core
    settings = get_settings()
    judge_model = args.judge_model or settings.judge_model
    # Fail before spending anything: an unpriced model would record $0 costs and
    # silently corrupt the cost columns this evaluation publishes.
    require_pricing([*settings.eval_models_list, judge_model])
    staleness = check_ground_truth_freshness(GROUND_TRUTH_CSV, settings)
    if staleness:
        print(staleness)
    gt = load_sample(args.n, args.seed)
    n_configs = len(settings.eval_models_list) * len(PROMPT_VERSIONS)
    print(f"[eval-rag] {len(gt)} questions x {n_configs} configs (judge={judge_model})")
    results = run(
        gt, settings, judge_model=judge_model,
        log_to_db=not args.no_log_db, workers=args.workers,
    )
    if not results:
        raise SystemExit("[eval-rag] no configs completed — see the errors above")
    if len(results) < n_configs:
        print(f"[eval-rag] NOTE: {n_configs - len(results)}/{n_configs} configs were skipped")
    write_outputs(results)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eval_rag", description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="questions to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judge-model", default=None,
        help="override the judge model (default: JUDGE_MODEL from config)",
    )
    parser.add_argument("--no-log-db", action="store_true", help="don't seed monitoring DB")
    parser.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return parser.parse_args()


if __name__ == "__main__":
    main()
