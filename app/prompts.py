"""Prompts (SPEC §8).

Two selectable system-prompt variants (``PROMPT_VERSION``): ``v1`` plain and
``v2`` with a few-shot example and a stricter citation instruction. Both require
grounded, cited answers. The context block renders each chunk under a citation
header so the model can cite ``[doc_title, para. X]``.
"""

from __future__ import annotations

from app.retrieval import RetrievedChunk

_BASE_RULES = (
    "You are an assistant for credit-risk analysts working with EU prudential "
    "regulation (IRB, definition of default, loan origination).\n"
    "Rules:\n"
    "- Answer ONLY from the provided context. Do not use outside knowledge.\n"
    "- If the context is insufficient, say so explicitly. Never guess.\n"
    "- Cite the source inline after each claim as [doc_title, para. X]."
)

SYSTEM_V1 = _BASE_RULES

SYSTEM_V2 = (
    _BASE_RULES
    + "\n- Every sentence that makes a factual claim MUST end with at least one "
    "citation in the exact form [doc_title, para. X]. Combine multiple sources "
    "as [doc_title, para. X; doc_title, para. Y].\n\n"
    "Example:\n"
    "Context:\n"
    "[EBA Guidelines on PD/LGD estimation (EBA/GL/2017/16), para. 82] "
    "Institutions shall apply an appropriate margin of conservatism to account "
    "for estimation error.\n"
    "Question: What must institutions do about estimation error in LGD?\n"
    "Answer: Institutions must apply an appropriate margin of conservatism to "
    "account for estimation error [EBA Guidelines on PD/LGD estimation "
    "(EBA/GL/2017/16), para. 82]."
)

_SYSTEM_BY_VERSION = {"v1": SYSTEM_V1, "v2": SYSTEM_V2}


def system_prompt(version: str) -> str:
    """Return the system prompt for the requested version."""
    if version not in _SYSTEM_BY_VERSION:
        raise ValueError(f"unknown prompt version: {version!r}")
    return _SYSTEM_BY_VERSION[version]


def citation_header(chunk_meta: RetrievedChunk) -> str:
    """The bracketed citation header for a chunk, e.g. '[Title, para. 82]'."""
    chunk = chunk_meta.chunk
    paras = ", ".join(chunk.para_ids) if chunk.para_ids else "n/a"
    return f"[{chunk.doc_title}, para. {paras}]"


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into a context block with citation headers."""
    blocks = [f"{citation_header(rc)}\n{rc.chunk.text}" for rc in chunks]
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    version: str,
    history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Assemble the chat messages for the answer call.

    `history` (prior user/assistant turns) is inserted before the current
    question so follow-ups can resolve references; grounding still comes only
    from the freshly retrieved context.
    """
    context = build_context(chunks)
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline citations."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt(version)}]
    messages.extend(safe_history(history))
    messages.append({"role": "user", "content": user})
    return messages


_HISTORY_ROLES = frozenset({"user", "assistant"})


def safe_history(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    """Keep only well-formed user/assistant turns.

    Never lets a caller-supplied ``system`` (or other) role reach the model — a
    prompt-injection defense, since history is client-controlled at the API.
    """
    safe: list[dict[str, str]] = []
    for message in history or []:
        role, content = message.get("role"), message.get("content")
        if role in _HISTORY_ROLES and isinstance(content, str):
            safe.append({"role": role, "content": content})
    return safe
