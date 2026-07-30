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
# A verdict we could not read. NOT one of RELEVANCE_LABELS: it is the absence of
# a judgement, not a negative one. Defaulting an unreadable response to
# NON_RELEVANT (as this module used to) silently pushes every config's score
# down, and hardest for whichever config makes the judge most verbose — a bias
# that looks exactly like a quality difference.
PARSE_ERROR = "PARSE_ERROR"

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
# One retry is worth it: a truncated or malformed judgement is usually transient,
# and dropping the question instead would shrink the sample unevenly.
_MAX_ATTEMPTS = 2
# Generous relative to the ~30 tokens the JSON needs. The judge is also asked for
# a short free-text reason, and a reasoning-model judge spends part of this
# budget thinking before emitting anything — running out mid-JSON is the failure
# this guards against. The prompt caps the reason to keep the cost negligible.
_MAX_TOKENS = 600

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
    '"reason": "..."}. Keep "reason" to one short sentence (max 25 words) and '
    "put it last, so the verdict fields are never cut off."
)


@dataclass(frozen=True)
class JudgeResult:
    relevance: str
    citations_supported: bool
    reason: str
    cost_usd: float = 0.0

    @property
    def parsed(self) -> bool:
        """True if the judge returned a verdict we could actually read."""
        return self.relevance in RELEVANCE_LABELS


def parse_judge(text: str) -> JudgeResult:
    """Parse the judge's JSON response.

    Returns ``relevance=PARSE_ERROR`` when the response is missing, malformed or
    carries an unrecognised label — never a real verdict. Callers must treat that
    as "not judged" and exclude it from rates, not fold it into NON_RELEVANT.
    """
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
        return JudgeResult(
            relevance=PARSE_ERROR, citations_supported=False, reason=_snippet(text)
        )
    return JudgeResult(
        relevance=label,
        citations_supported=bool(data.get("citations_supported", False)),
        reason=str(data.get("reason", "")).strip(),
    )


def _snippet(text: str, limit: int = 200) -> str:
    """A short excerpt of an unparseable response, for diagnosis."""
    cleaned = " ".join(text.split())
    return cleaned[:limit] if len(cleaned) <= limit else cleaned[:limit] + "…"


def judge_answer(
    question: str,
    reference: str,
    answer_text: str,
    sources_text: str,
    *,
    model: str,
) -> JudgeResult:
    """Judge one answer, retrying once if the response can't be parsed.

    Returns a ``PARSE_ERROR`` verdict if both attempts fail, with the cost of
    every attempt so accounting stays honest. Never invents a verdict.
    """
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE:\n{reference}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        f"SOURCES:\n{sources_text}"
    )
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]
    cost = 0.0
    parsed = JudgeResult(PARSE_ERROR, False, "no attempt made")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        result: LLMResult = complete(
            messages, model=model, temperature=0.0, max_tokens=_MAX_TOKENS
        )
        cost += result.cost_usd
        parsed = parse_judge(result.text)
        if parsed.parsed:
            break
        # Say why, so a systematic failure (e.g. a too-small budget for a
        # reasoning judge) is diagnosable instead of just a lower score.
        cause = "truncated at the token cap" if result.truncated else "unparseable response"
        print(
            f"[judge] attempt {attempt}/{_MAX_ATTEMPTS} failed ({cause}, "
            f"finish_reason={result.finish_reason!r}, out={result.tokens_out} tok): "
            f"{parsed.reason}"
        )
    return JudgeResult(
        relevance=parsed.relevance,
        citations_supported=parsed.citations_supported,
        reason=parsed.reason,
        cost_usd=cost,
    )
