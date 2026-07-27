"""LLM-as-a-judge for RAG evaluation (SPEC §11.3).

One judge call per answer returns both a 3-class relevance label and whether the
answer's citations are supported by the sources it was given. The response parser
is pure so it can be unit-tested without an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.providers import LLMResult, complete

RELEVANCE_LABELS = ("RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT")

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM = (
    "You are a strict evaluator for a RAG system over EU credit-risk regulation. "
    "You are given a QUESTION, a REFERENCE passage (the ground-truth source), an "
    "ANSWER, and the SOURCES the answer was shown (each labelled with its "
    "citation header). Judge two things independently:\n"
    "1) relevance — compare the ANSWER to the REFERENCE: RELEVANT (correctly and "
    "fully answers), PARTLY_RELEVANT (partially correct or incomplete), or "
    "NON_RELEVANT (wrong or unsupported).\n"
    "2) citations_supported — judge ONLY against the SOURCES (ignore the "
    "reference here): true if each bracketed [doc, para. X] citation matches a "
    "provided source whose text supports the accompanying claim.\n"
    'Return ONLY JSON: {"relevance": "...", "citations_supported": true|false, '
    '"reason": "..."}.'
)


@dataclass(frozen=True)
class JudgeResult:
    relevance: str
    citations_supported: bool
    reason: str
    cost_usd: float = 0.0


def parse_judge(text: str) -> JudgeResult:
    """Parse the judge's JSON response, defaulting safely on malformed output."""
    match = _JSON_OBJ_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}
    label = str(data.get("relevance", "")).strip().upper()
    if label not in RELEVANCE_LABELS:
        label = "NON_RELEVANT"
    return JudgeResult(
        relevance=label,
        citations_supported=bool(data.get("citations_supported", False)),
        reason=str(data.get("reason", "")).strip(),
    )


def judge_answer(
    question: str,
    reference: str,
    answer_text: str,
    sources_text: str,
    *,
    model: str,
) -> JudgeResult:
    """Run one judge call and return the parsed result with its cost."""
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE:\n{reference}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        f"SOURCES:\n{sources_text}"
    )
    result: LLMResult = complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        model=model,
        temperature=0.0,
        max_tokens=200,
    )
    parsed = parse_judge(result.text)
    return JudgeResult(
        relevance=parsed.relevance,
        citations_supported=parsed.citations_supported,
        reason=parsed.reason,
        cost_usd=result.cost_usd,
    )
