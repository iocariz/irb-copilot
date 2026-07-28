"""RAG orchestration (SPEC §8).

``answer(question, doc_ids=None)`` runs: rewrite (if enabled) -> retrieve ->
build prompt -> LLM -> parse citations -> return an ``Answer``. Token counts and
cost aggregate the rewrite and answer LLM calls; latency is end-to-end.

``Answer`` is a pydantic model (a boundary type — it is serialized by the API
and consumed by the UI), superseding the plain dataclass in the spec.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.prompts import build_messages
from app.providers import LLMResult, complete, stream_complete
from app.retrieval import RetrievedChunk, Retriever, get_retriever
from app.rewrite import RewriteResult, condense_query, rewrite_query

# One bracketed citation group, and its "title, para. X" internals.
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
# Accept the prompt's "para." plus the forms a model sometimes emits anyway
# ("paras", "paragraph(s)"), so a valid citation isn't silently dropped.
_CITE_RE = re.compile(
    r"^(?P<title>.+?),\s*(?:paragraphs?|paras?)\.?\s*(?P<paras>.+)$", re.IGNORECASE
)
_PARA_SPLIT_RE = re.compile(r"[,;]")
# Title normalization for grounding: LLMs often drop dates/parentheticals.
_PAREN_RE = re.compile(r"\([^)]*\)")
_WS_RE = re.compile(r"\s+")
# Stable regulatory identifiers that survive paraphrase (EBA/GL/2017/16, BCBS d424).
_DOC_CODE_RE = re.compile(r"\b(?:EBA/GL/\d{4}/\d+|BCBS\s*d?\d+)\b", re.IGNORECASE)
# Token-containment fallback: one title's content words are (mostly) a subset of
# the other's. Requires enough shared words so generic 2-word titles (e.g. bare
# "EBA Guidelines") do not match every corpus document.
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Only >2-char fillers matter (shorter words are dropped by the length filter).
_TITLE_STOPWORDS = frozenset({"the", "and", "for", "with", "under", "from", "that", "this"})
_TITLE_MIN_SHARED_TOKENS = 3
_TITLE_MIN_CONTAINMENT = 0.8


class Citation(BaseModel):
    """A citation parsed from the answer text."""

    text: str
    doc_title: str
    paras: str


class SourceChunk(BaseModel):
    """A retrieved chunk exposed to the API/UI, with its retrieval score."""

    chunk_id: str
    doc_id: str
    doc_title: str
    section_path: list[str]
    para_ids: list[str]
    pages: list[int]
    score: float
    text: str


class Answer(BaseModel):
    """A grounded, cited answer plus retrieval + accounting metadata."""

    text: str
    citations: list[Citation]
    chunks_used: list[SourceChunk]
    question: str
    rewritten_query: str
    retrieval_mode: str
    prompt_version: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    latency_ms: int
    used_rewrite: bool = Field(default=False)
    # Answer-time self-check: are all citations backed by a retrieved source?
    citations_grounded: bool = Field(default=True)
    ungrounded_citations: list[str] = Field(default_factory=list)


def parse_citations(text: str) -> list[Citation]:
    """Extract unique '[doc_title, para. X]' citations from answer text (pure)."""
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for bracket in _BRACKET_RE.finditer(text):
        for part in bracket.group(1).split(";"):
            part = part.strip()
            match = _CITE_RE.match(part)
            if not match:
                continue
            title, paras = match.group("title").strip(), match.group("paras").strip()
            if (title, paras) not in seen:
                seen.add((title, paras))
                citations.append(Citation(text=part, doc_title=title, paras=paras))
    return citations


def _normalize_title(title: str) -> str:
    """Lowercase title with parentheticals removed and whitespace collapsed."""
    return _WS_RE.sub(" ", _PAREN_RE.sub(" ", title)).strip().casefold()


def _doc_codes(title: str) -> set[str]:
    """Regulatory codes embedded in a title (e.g. ``eba/gl/2017/16``)."""
    return {m.group(0).casefold().replace(" ", "") for m in _DOC_CODE_RE.finditer(title)}


def _title_tokens(normalized_title: str) -> set[str]:
    """Distinctive content words of an already-normalized title."""
    return {
        t for t in _TITLE_TOKEN_RE.findall(normalized_title)
        if len(t) > 2 and t not in _TITLE_STOPWORDS
    }


def titles_match(cited: str, available: str) -> bool:
    """True if a cited title refers to the same document as a retrieved title.

    Cascade, most-precise first: exact (case-insensitive) → shared regulatory
    code (EBA/GL/2017/16, BCBS d424) → normalized equality → token-containment
    (one title's content words are ≥80% a subset of the other's, with at least
    3 shared words). The last rule handles front/middle shortenings while the
    minimum-shared-words guard rejects generic titles like bare ``EBA Guidelines``.
    """
    if cited.strip().casefold() == available.strip().casefold():
        return True
    cited_codes, available_codes = _doc_codes(cited), _doc_codes(available)
    if cited_codes and available_codes and cited_codes & available_codes:
        return True
    a, b = _normalize_title(cited), _normalize_title(available)
    if not a or not b:
        return False
    if a == b:
        return True
    a_tokens, b_tokens = _title_tokens(a), _title_tokens(b)
    if not a_tokens or not b_tokens:
        return False
    shared = len(a_tokens & b_tokens)
    smaller = min(len(a_tokens), len(b_tokens))
    return shared >= _TITLE_MIN_SHARED_TOKENS and shared / smaller >= _TITLE_MIN_CONTAINMENT


def check_citation_grounding(
    citations: list[Citation], chunks_used: list[SourceChunk]
) -> list[str]:
    """Return the texts of citations NOT backed by a retrieved source (pure).

    A citation is grounded only if a retrieved document matches its title (exact
    or fuzzy — see ``titles_match``) and **every** cited paragraph number was
    actually retrieved. Requiring all paragraphs — not just one — means a partly
    hallucinated citation (e.g. ``para. 82, 99`` when only 82 was retrieved) is
    flagged rather than laundered by the one real paragraph.
    """
    available: dict[str, set[str]] = {}
    for chunk in chunks_used:
        available.setdefault(chunk.doc_title, set()).update(chunk.para_ids)
    ungrounded: list[str] = []
    for citation in citations:
        cited = {p.strip() for p in _PARA_SPLIT_RE.split(citation.paras) if p.strip()}
        # Paragraphs available under any document whose title matches the citation.
        covered: set[str] = set()
        for title, paras in available.items():
            if titles_match(citation.doc_title, title):
                covered |= paras
        if not cited or not cited.issubset(covered):
            ungrounded.append(citation.text)
    return ungrounded


def _to_source(rc: RetrievedChunk) -> SourceChunk:
    c = rc.chunk
    return SourceChunk(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        doc_title=c.doc_title,
        section_path=c.section_path,
        para_ids=c.para_ids,
        pages=c.pages,
        score=rc.score,
        text=c.text,
    )


def _prepare(
    question: str,
    doc_ids: list[str] | None,
    settings: Settings,
    retriever: Retriever,
    history: list[dict[str, str]] | None,
) -> tuple[RewriteResult, list[SourceChunk], list[dict[str, str]]]:
    """Shared front half: resolve the retrieval query, retrieve, build messages.

    For a follow-up (history present) the query is condensed to a standalone form
    using that history, so retrieval isn't run on a referent-free question; a
    first-turn question uses the plain rewrite (acronym expansion, off by default).
    """
    if history and settings.condense_history:
        rewrite = condense_query(question, history, settings)
    else:
        rewrite = rewrite_query(question, settings)
    chunks = retriever.search(
        rewrite.rewritten,
        mode=settings.retrieval_mode,
        top_k=settings.top_k,
        doc_ids=doc_ids,
    )
    messages = build_messages(
        question, chunks, version=settings.prompt_version, history=history
    )
    return rewrite, [_to_source(rc) for rc in chunks], messages


def _assemble(
    question: str,
    rewrite: RewriteResult,
    sources: list[SourceChunk],
    llm: LLMResult,
    settings: Settings,
    started: float,
) -> Answer:
    """Shared back half: parse citations, self-check, assemble the Answer."""
    citations = parse_citations(llm.text)
    ungrounded = check_citation_grounding(citations, sources)
    return Answer(
        text=llm.text,
        citations=citations,
        chunks_used=sources,
        citations_grounded=not ungrounded,
        ungrounded_citations=ungrounded,
        question=question,
        rewritten_query=rewrite.rewritten,
        retrieval_mode=settings.retrieval_mode,
        prompt_version=settings.prompt_version,
        model=llm.model,
        tokens_in=rewrite.tokens_in + llm.tokens_in,
        tokens_out=rewrite.tokens_out + llm.tokens_out,
        cost_usd=round(rewrite.cost_usd + llm.cost_usd, 8),
        latency_ms=int((time.perf_counter() - started) * 1000),
        used_rewrite=rewrite.used_llm,
    )


def answer(
    question: str,
    doc_ids: list[str] | None = None,
    *,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    history: list[dict[str, str]] | None = None,
) -> Answer:
    """Run the full RAG flow and return a cited Answer."""
    settings = settings or get_settings()
    retriever = retriever or get_retriever()
    started = time.perf_counter()
    rewrite, sources, messages = _prepare(question, doc_ids, settings, retriever, history)
    llm = complete(messages, model=settings.llm_model, max_tokens=settings.max_answer_tokens)
    return _assemble(question, rewrite, sources, llm, settings, started)


def answer_stream(
    question: str,
    doc_ids: list[str] | None = None,
    *,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    history: list[dict[str, str]] | None = None,
) -> Iterator[tuple[str, object]]:
    """Streaming RAG flow.

    Yields ("sources", list[SourceChunk]) once retrieval is done, then
    ("token", str) for each generated delta, then ("answer", Answer) at the end.
    """
    settings = settings or get_settings()
    retriever = retriever or get_retriever()
    started = time.perf_counter()
    rewrite, sources, messages = _prepare(question, doc_ids, settings, retriever, history)
    yield ("sources", sources)

    stream = stream_complete(
        messages, model=settings.llm_model, max_tokens=settings.max_answer_tokens
    )
    llm: LLMResult | None = None
    while True:
        try:
            delta = next(stream)
        except StopIteration as stop:
            llm = stop.value
            break
        yield ("token", delta)

    if llm is None:  # only reachable if the stream ended without a final result
        raise RuntimeError("stream_complete ended without returning an LLMResult")
    yield ("answer", _assemble(question, rewrite, sources, llm, settings, started))
