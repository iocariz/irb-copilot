"""Ground-truth generation (SPEC §11.1).

    python -m evaluation.generate_ground_truth --n 200 --seed 42

Samples ~N structure chunks (stratified by document); for each, an LLM writes a
handful of questions a risk modeler might ask whose answer is in that chunk.
Writes evaluation/ground_truth.csv (question, doc_id, chunk_id, para_ids) and
prints 20 random rows for manual spot-checking. Reproducible via --seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.providers import complete
from evaluation.corpus import load_chunks
from evaluation.metrics import lexical_overlap
from evaluation.sampling import stratified_sample
from ingestion.models import Chunk

GROUND_TRUTH_CSV = PROJECT_ROOT / "evaluation" / "ground_truth.csv"

_SYSTEM = (
    "You generate evaluation questions for a retrieval system over EU credit-risk "
    "regulation. Given one passage, write questions a credit-risk analyst might ask "
    "whose answer is fully contained in THAT passage. Questions must be answerable "
    "from the passage alone, specific, and self-contained (no 'this passage'). "
    "Return ONLY a JSON array of question strings."
)

# "hard" style: paraphrase away the passage vocabulary to reduce lexical bias,
# so the eval better reflects real user queries (and tests semantic retrieval).
_SYSTEM_HARD = (
    "You generate DE-BIASED evaluation questions for a retrieval system over EU "
    "credit-risk regulation. Given one passage, write questions a credit-risk "
    "analyst might ask whose answer is in THAT passage — but PARAPHRASE heavily: "
    "avoid reusing the passage's distinctive words and phrases, prefer synonyms "
    "and plain practitioner language, and never quote the passage. Questions must "
    "still be answerable from the passage alone and self-contained. Return ONLY a "
    "JSON array of question strings."
)

_SYSTEMS = {"standard": _SYSTEM, "hard": _SYSTEM_HARD}

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def ground_truth_path(style: str) -> Path:
    """Output CSV path for a ground-truth style (standard -> ground_truth.csv)."""
    if style == "standard":
        return GROUND_TRUTH_CSV
    return PROJECT_ROOT / "evaluation" / f"ground_truth_{style}.csv"


def parse_questions(response_text: str, n: int) -> list[str]:
    """Parse the LLM response into up to `n` clean question strings (pure)."""
    match = _JSON_ARRAY_RE.search(response_text)
    questions: list[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            questions = [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            questions = []
    if not questions:  # fallback: numbered / bulleted lines
        for line in response_text.splitlines():
            cleaned = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)
    return questions[:n]


def generate_questions(chunk: Chunk, n: int, model: str, system: str = _SYSTEM) -> list[str]:
    """Ask the LLM for `n` questions answerable from `chunk`."""
    user = (
        f"Passage (from {chunk.doc_title}, para. {', '.join(chunk.para_ids)}):\n"
        f"{chunk.text}\n\n"
        f"Write {n} such questions as a JSON array of strings."
    )
    result = complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=0.3,
        max_tokens=400,
    )
    return parse_questions(result.text, n)


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    system = _SYSTEMS[args.style]
    out_path = ground_truth_path(args.style)
    chunks = load_chunks("structure", settings)
    sample = stratified_sample(chunks, args.n, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    print(
        f"[gt] style={args.style}; {len(sample)} chunks x {args.questions} questions "
        f"({args.workers} workers)"
    )

    # Generate concurrently (I/O-bound API calls); keep results in sample order.
    per_chunk: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(generate_questions, chunk, args.questions, settings.llm_model, system): i
            for i, chunk in enumerate(sample)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            per_chunk[futures[future]] = future.result()
            if done % 25 == 0:
                print(f"[gt] {done}/{len(sample)} chunks done")

    rows: list[dict[str, str]] = []
    overlaps: list[float] = []
    for i, chunk in enumerate(sample):
        for question in per_chunk.get(i, []):
            rows.append(
                {
                    "question": question,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "para_ids": ",".join(chunk.para_ids),
                }
            )
            overlaps.append(lexical_overlap(question, chunk.text))

    _write_csv(rows, out_path)
    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
    print(f"[gt] wrote {len(rows)} questions to {out_path}")
    print(f"[gt] mean question↔chunk lexical overlap: {mean_overlap:.3f} (lower = less bias)")
    _print_samples(rows, seed=args.seed)


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["question", "doc_id", "chunk_id", "para_ids"])
        writer.writeheader()
        writer.writerows(rows)


def _print_samples(rows: list[dict[str, str]], *, seed: int, k: int = 20) -> None:
    print(f"\n--- {min(k, len(rows))} random samples for spot-checking ---")
    for row in random.Random(seed).sample(rows, min(k, len(rows))):
        print(f"[{row['doc_id']} para {row['para_ids']}] {row['question']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="generate_ground_truth", description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="chunks to sample")
    parser.add_argument("--questions", type=int, default=4, help="questions per chunk")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap chunks (cheap runs)")
    parser.add_argument(
        "--style", choices=("standard", "hard"), default="standard",
        help="hard = paraphrased, de-biased questions (lower lexical overlap)",
    )
    parser.add_argument("--workers", type=int, default=8, help="concurrent API calls")
    return parser.parse_args()


if __name__ == "__main__":
    main()
