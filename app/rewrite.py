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

# All-caps tokens (2–6 letters, optional digit) — used to find UNKNOWN acronyms.
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}[0-9]?)\b")


def _glossary_regex(glossary: dict[str, str]) -> re.Pattern[str]:
    """Match any glossary term case-insensitively (longest key first, so ELBE
    wins over EL). Keyed off the glossary so mixed-case keys like MoC/DoD work."""
    keys = sorted((re.escape(k) for k in glossary), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(keys) + r")\b", re.IGNORECASE)


_GLOSSARY_RE = _glossary_regex(GLOSSARY)

_REWRITE_SYSTEM = (
    "You rewrite a credit-risk analyst's question into a concise search query "
    "for retrieving passages from EU prudential regulation. Expand remaining "
    "acronyms, drop conversational filler, and keep key domain terms. Return "
    "ONLY the rewritten query, no preamble."
)

_CONDENSE_SYSTEM = (
    "Given the conversation so far and a follow-up question, rewrite the follow-up "
    "into a STANDALONE search query for retrieving passages from EU prudential "
    "regulation. Resolve references (it, that, this guideline, the next paragraph) "
    "to the concrete documents, paragraph numbers, and terms they point to, and "
    "keep exact regulatory wording. Do NOT answer. Return ONLY the query."
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
    """Append 'ACRONYM (expansion)' for each known acronym found (pure).

    Case-insensitive, so mixed-case keys (MoC, DoD) and any user casing
    (MoC / MOC / moc) all resolve to the same glossary entry.
    """
    if not glossary:
        return text
    regex = _GLOSSARY_RE if glossary is GLOSSARY else _glossary_regex(glossary)
    canonical = {key.upper(): value for key, value in glossary.items()}
    seen: set[str] = set()
    additions: list[str] = []
    for match in regex.finditer(text):
        token = match.group(1)
        key = token.upper()
        if key not in seen:
            seen.add(key)
            additions.append(f"{token} ({canonical[key]})")
    if not additions:
        return text
    return f"{text} [{'; '.join(additions)}]"


def unknown_acronyms(text: str, glossary: dict[str, str] = GLOSSARY) -> list[str]:
    """All-caps tokens not in the glossary (candidate unknown terms), compared
    case-insensitively so a known acronym in any casing isn't flagged."""
    known = {key.upper() for key in glossary}
    return [m.group(1) for m in _ACRONYM_RE.finditer(text) if m.group(1).upper() not in known]


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


def condense_query(
    question: str,
    history: list[dict[str, str]],
    settings: Settings | None = None,
) -> RewriteResult:
    """Rewrite a follow-up into a standalone retrieval query using chat history.

    Retrieval otherwise runs on the bare follow-up, so references like "the next
    paragraph" would search with a weak, context-free query. Only meaningful when
    there is history; falls back to the plain rewrite otherwise.
    """
    from app.prompts import safe_history  # local import avoids an import cycle

    settings = settings or get_settings()
    turns = safe_history(history)
    if not turns:
        return rewrite_query(question, settings)

    conversation = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    user = f"Conversation:\n{conversation}\n\nFollow-up: {question}\n\nStandalone query:"
    result = complete(
        [
            {"role": "system", "content": _CONDENSE_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=settings.llm_model,
        temperature=0.0,
        max_tokens=128,
    )
    return RewriteResult(
        original=question,
        rewritten=result.text.strip() or question,
        used_llm=True,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_usd=result.cost_usd,
    )
