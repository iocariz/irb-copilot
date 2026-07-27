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

from app.config import PROJECT_ROOT, get_settings
from app.providers import complete
from evaluation.corpus import load_chunks
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

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


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


def generate_questions(chunk: Chunk, n: int, model: str) -> list[str]:
    """Ask the LLM for `n` questions answerable from `chunk`."""
    user = (
        f"Passage (from {chunk.doc_title}, para. {', '.join(chunk.para_ids)}):\n"
        f"{chunk.text}\n\n"
        f"Write {n} such questions as a JSON array of strings."
    )
    result = complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        model=model,
        temperature=0.3,
        max_tokens=400,
    )
    return parse_questions(result.text, n)


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    chunks = load_chunks("structure", settings)
    sample = stratified_sample(chunks, args.n, seed=args.seed)
    if args.limit:
        sample = sample[: args.limit]
    print(f"[gt] sampling {len(sample)} chunks; generating {args.questions} questions each")

    rows: list[dict[str, str]] = []
    for i, chunk in enumerate(sample, start=1):
        for question in generate_questions(chunk, args.questions, settings.llm_model):
            rows.append(
                {
                    "question": question,
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    "para_ids": ",".join(chunk.para_ids),
                }
            )
        if i % 25 == 0:
            print(f"[gt] {i}/{len(sample)} chunks -> {len(rows)} questions")

    _write_csv(rows)
    print(f"[gt] wrote {len(rows)} questions to {GROUND_TRUTH_CSV}")
    _print_samples(rows, seed=args.seed)


def _write_csv(rows: list[dict[str, str]]) -> None:
    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with GROUND_TRUTH_CSV.open("w", encoding="utf-8", newline="") as fh:
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
    return parser.parse_args()


if __name__ == "__main__":
    main()
