"""Query rewriting (SPEC §7).

Two steps before retrieval, toggleable via ``ENABLE_REWRITE``:

1. expand domain acronyms from a hardcoded glossary (deterministic, no cost);
2. if unknown all-caps terms remain, or to reformulate conversational phrasing
   into a search query, make one cheap LLM call.

The glossary step is a pure function so it is unit-testable without any LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.providers import complete

# Domain acronyms for IRB / definition of default / loan origination.
GLOSSARY: dict[str, str] = {
    "PD": "probability of default",
    "LGD": "loss given default",
    "EAD": "exposure at default",
    "CCF": "credit conversion factor",
    "EL": "expected loss",
    "MoC": "margin of conservatism",
    "DoD": "definition of default",
    "CRR": "Capital Requirements Regulation",
    "CRD": "Capital Requirements Directive",
    "RDS": "reference data set",
    "IRB": "internal ratings-based approach",
    "RWA": "risk-weighted assets",
    "DPD": "days past due",
    "UTP": "unlikeliness to pay",
    "ELBE": "expected loss best estimate",
    "RWEA": "risk-weighted exposure amounts",
}

# All-caps tokens (2–6 letters, optional digits) that look like acronyms.
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}[0-9]?)\b")

_REWRITE_SYSTEM = (
    "You rewrite a credit-risk analyst's question into a concise search query "
    "for retrieving passages from EU prudential regulation. Expand remaining "
    "acronyms, drop conversational filler, and keep key domain terms. Return "
    "ONLY the rewritten query, no preamble."
)


@dataclass(frozen=True)
class RewriteResult:
    """Original and rewritten query, whether the LLM was used, and its cost."""

    original: str
    rewritten: str
    used_llm: bool
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


def expand_acronyms(text: str, glossary: dict[str, str] = GLOSSARY) -> str:
    """Append 'ACRONYM (expansion)' for each known acronym found (pure)."""
    seen: set[str] = set()
    additions: list[str] = []
    for match in _ACRONYM_RE.finditer(text):
        token = match.group(1)
        if token in glossary and token not in seen:
            seen.add(token)
            additions.append(f"{token} ({glossary[token]})")
    if not additions:
        return text
    return f"{text} [{'; '.join(additions)}]"


def unknown_acronyms(text: str, glossary: dict[str, str] = GLOSSARY) -> list[str]:
    """All-caps tokens that are not in the glossary (candidate unknown terms)."""
    return [
        m.group(1)
        for m in _ACRONYM_RE.finditer(text)
        if m.group(1) not in glossary
    ]


def rewrite_query(question: str, settings: Settings | None = None) -> RewriteResult:
    """Expand acronyms; call the LLM to reformulate when rewriting is enabled."""
    settings = settings or get_settings()
    expanded = expand_acronyms(question)

    if not settings.enable_rewrite:
        return RewriteResult(original=question, rewritten=expanded, used_llm=False)

    result = complete(
        [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": expanded},
        ],
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=128,
    )
    rewritten = result.text.strip() or expanded
    return RewriteResult(
        original=question,
        rewritten=rewritten,
        used_llm=True,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )
