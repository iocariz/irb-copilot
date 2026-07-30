"""Streamlit front end (SPEC §10).

A single page: ask a question (optionally filtered to specific documents), watch
the cited answer stream in, expand the source snippets, rate it 👍/👎, and ask
follow-ups (a short conversation history is kept). Calls the FastAPI backend's
streaming endpoint; model/cost/latency of the last answer are in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/ui.py` puts app/ (not the repo root) on sys.path, so
# `import app.*` would fail; add the repo root before importing app modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import httpx
import streamlit as st
import yaml

from app.config import get_settings

settings = get_settings()
_TIMEOUT = httpx.Timeout(120.0)
_HISTORY_TURNS = 3  # prior turns sent as follow-up context


def _headers() -> dict[str, str]:
    """Attach the API key when the backend requires one."""
    return {"X-API-Key": settings.api_key} if settings.api_key else {}


def load_documents() -> dict[str, str]:
    """Return {doc_id: title} from data/sources.yaml for the document filter."""
    path = settings.data_path / "sources.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {d["id"]: d["title"] for d in raw.get("documents", [])}


def sse_ask(question: str, doc_ids: list[str], history: list[dict]):
    """Yield (event, payload) parsed from the /ask/stream SSE response."""
    payload = {"question": question, "doc_ids": doc_ids or None, "history": history or None}
    with httpx.stream(
        "POST", f"{settings.api_url}/ask/stream", json=payload, headers=_headers(), timeout=_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        event = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                yield event, json.loads(line[len("data: ") :])


def send_feedback(answer_id: str, thumbs: str, comment: str | None = None) -> None:
    httpx.post(
        f"{settings.api_url}/feedback",
        json={"answer_id": answer_id, "thumbs": thumbs, "comment": comment},
        headers=_headers(),
        timeout=_TIMEOUT,
    ).raise_for_status()


def format_pages(pages: list[int]) -> str:
    """Render page numbers for a citation (pure).

    Was interpolating the list directly, which printed Python syntax: "(p. [12,
    13])". Contiguous runs collapse to a range, which is how a page reference is
    normally written.
    """
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p. {pages[0]}"
    if pages[-1] - pages[0] == len(pages) - 1:  # contiguous
        return f"pp. {pages[0]}–{pages[-1]}"
    return "pp. " + ", ".join(str(p) for p in pages)


def source_header(src: dict) -> str:
    """Label for a source expander (pure).

    Not every chunk has a paragraph number: annexes and tables carry none by
    design, since inventing one is what put 48% of the corpus under a wrong
    citation anchor. 24% of chunks are unnumbered, and this used to render them
    as a bare "— para.  " with nothing after it. Those fall back to their section
    path, which is the honest anchor for that content.
    """
    section = (src.get("section_path") or [""])[-1].strip()
    if src.get("para_ids"):
        anchor = "para. " + ", ".join(src["para_ids"])
    elif section and section.lower() not in src["doc_title"].lower():
        # Skip a top-level heading that merely repeats the document title, which
        # would render as "Basel III … — § Basel III …".
        anchor = "§ " + section
    else:
        anchor = "unnumbered"
    pages = format_pages(src.get("pages") or [])
    return f"{src['doc_title']} — {anchor}" + (f" · {pages}" if pages else "")


def _feedback_state_key(answer_id: str) -> str:
    return f"feedback_sent::{answer_id}"


def render_details(ans: dict) -> None:
    """Citations, grounding warning, source snippets, and feedback buttons."""
    if ans.get("citations"):
        st.caption("Citations: " + " · ".join(c["text"] for c in ans["citations"]))
    else:
        # The whole design rests on every claim carrying a source, so silence here
        # is the failure this tool exists to prevent. Stated neutrally, because an
        # answer that correctly declines ("the context does not cover this") also
        # has no citations and is behaving properly.
        st.warning(
            "⚠ This answer cites no sources. Either the model declined to answer "
            "from the corpus, or it answered without grounding — check before "
            "relying on it."
        )
    if ans.get("truncated"):
        # Otherwise the answer just stops mid-sentence and reads as complete.
        st.warning(
            "⚠ This answer hit the output-token limit and is cut off. Ask a "
            "narrower question, or raise `MAX_ANSWER_TOKENS`."
        )
    if ans.get("ungrounded_citations"):
        st.warning(
            "⚠ These citations were not found in the retrieved sources "
            "(possible hallucination): " + "; ".join(ans["ungrounded_citations"])
        )
    _render_search_query(ans)
    _render_sources(ans["chunks_used"], ans.get("retrieval_mode"))
    _render_feedback(ans)


def _render_search_query(ans: dict) -> None:
    """Show the query retrieval actually ran on, when it isn't the question.

    A follow-up is condensed into a standalone query before retrieval ("what
    about the materiality threshold?" becomes something quite different), and
    until now that rewrite was invisible. When the sources look wrong, this is
    the first thing worth checking.
    """
    searched = (ans.get("rewritten_query") or "").strip()
    if searched and searched != (ans.get("question") or "").strip():
        st.caption(f"🔍 Retrieved on: _{searched}_")


def format_score(score: float, mode: str | None) -> str:
    """Retrieval score, always labelled with the mode that produced it (pure).

    Scores are not comparable across modes: cosine similarity (0–1), BM25
    (unbounded), reciprocal rank fusion (~1/60 scale) and the cross-encoder's
    own scale all surface here — a top `hybrid_rerank` hit reads ~0.4 where a
    `hybrid` one reads ~0.016, for the same quality of match. Showing a bare
    number invites reading it as a confidence; naming the mode makes clear it is
    a raw score, and rank is the signal that survives the comparison.
    """
    return f"{mode} score {score:.3g}" if mode else f"score {score:.3g}"


def _render_sources(sources: list[dict], retrieval_mode: str | None = None) -> None:
    """Sources, cited ones first and already open.

    Verifying a claim is the actual task here, and it used to mean opening every
    collapsed source in turn to find which one backed a given citation. Sources
    the answer leaned on are marked, expanded, and listed first; the rest stay
    available but out of the way.

    Retrieval rank is shown explicitly *because* of that reordering — it used to
    be implicit in the order, and grouping by citation destroyed it. Rank is the
    mode-independent signal; the raw score is secondary.
    """
    ranked = list(enumerate(sources, start=1))
    cited = [(rank, s) for rank, s in ranked if s.get("cited_by")]
    uncited = [(rank, s) for rank, s in ranked if not s.get("cited_by")]
    st.markdown(f"#### Sources ({len(cited)} of {len(sources)} cited)")
    for rank, src in cited:
        with st.expander(f"✓ #{rank} " + source_header(src), expanded=True):
            st.caption("Backs: " + " · ".join(src["cited_by"]))
            _render_source_body(src, rank, len(sources), retrieval_mode)
    if uncited:
        st.caption(
            f"{len(uncited)} further source(s) were retrieved but not cited by the answer."
        )
        for rank, src in uncited:
            with st.expander(f"○ #{rank} " + source_header(src)):
                _render_source_body(src, rank, len(sources), retrieval_mode)


def _render_source_body(
    src: dict, rank: int, total: int, retrieval_mode: str | None
) -> None:
    if src.get("section_path"):
        st.caption(" › ".join(src["section_path"]))
    st.caption(f"rank {rank} of {total} · {format_score(src.get('score', 0.0), retrieval_mode)}")
    st.write(src["text"])


def _render_feedback(ans: dict) -> None:
    """Thumbs buttons, or a record of the rating already given.

    Each click used to POST a new `feedback` row, so a user clicking twice skewed
    the up/down ratio panel — which has the least data of any panel and so is the
    most distorted by duplicates.
    """
    answer_id = ans.get("answer_id")
    already = st.session_state.get(_feedback_state_key(answer_id)) if answer_id else None
    if already:
        st.caption(f"Feedback recorded: {'👍' if already == 'up' else '👎'} — thank you.")
        return
    col_up, col_down, _ = st.columns([1, 1, 6])
    if col_up.button("👍", key=f"up_{answer_id}"):
        _submit_feedback(ans, "up")
    if col_down.button("👎", key=f"down_{answer_id}"):
        _submit_feedback(ans, "down")


def render_answer(ans: dict) -> None:
    """Static (non-streaming) render of a stored answer."""
    st.markdown("### Answer")
    st.write(ans["text"])
    render_details(ans)


def _submit_feedback(ans: dict, thumbs: str) -> None:
    if not ans.get("answer_id"):
        st.warning("Answer was not logged; feedback unavailable.")
        return
    try:
        send_feedback(ans["answer_id"], thumbs)
    except Exception as exc:  # noqa: BLE001
        # Do NOT mark it as sent — the user should be able to retry.
        st.error(f"Could not send feedback: {exc}")
        return
    # Marked only after the POST succeeds, so the buttons are replaced by the
    # recorded rating and a second click cannot log a duplicate row.
    st.session_state[_feedback_state_key(ans["answer_id"])] = thumbs
    st.rerun()


def render_sidebar() -> None:
    st.sidebar.header("Last answer")
    ans = st.session_state.get("answer")
    if not ans:
        st.sidebar.caption("Ask a question to see stats.")
        return
    st.sidebar.metric("Model", ans["model"])
    st.sidebar.metric("Cost (USD)", f"${ans['cost_usd']:.5f}")
    st.sidebar.metric("Latency (ms)", ans["latency_ms"])
    st.sidebar.caption(
        f"mode: {ans['retrieval_mode']} · prompt: {ans['prompt_version']} · "
        f"tokens: {ans['tokens_in']}→{ans['tokens_out']}"
    )


def _run_streaming_ask(question: str, doc_ids: list[str]) -> None:
    history = st.session_state.get("history", [])
    st.markdown("### Answer")
    captured: dict[str, dict] = {}

    def tokens():
        try:
            for event, payload in sse_ask(question, doc_ids, history[-2 * _HISTORY_TURNS :]):
                if event == "token":
                    yield payload["text"]
                else:
                    captured[event] = payload
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")

    st.write_stream(tokens())
    if "error" in captured:
        st.error(captured["error"].get("detail", "The answer could not be generated."))
        return
    final = captured.get("done")
    if not final:
        return
    render_details(final)
    st.session_state["answer"] = final
    st.session_state["history"] = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": final["text"]},
    ]


def main() -> None:
    st.set_page_config(page_title="IRB Copilot", page_icon="📘")
    st.title("📘 IRB Copilot")
    st.caption(
        "Q&A over EU prudential regulation for credit-risk modeling. "
        "Answers cite their sources."
    )
    st.session_state.setdefault("history", [])

    docs = load_documents()
    selected_titles = st.multiselect(
        "Restrict to documents (optional)", options=list(docs.values())
    )
    title_to_id = {title: doc_id for doc_id, title in docs.items()}
    doc_ids = [title_to_id[t] for t in selected_titles]

    question = st.text_area(
        "Your question",
        placeholder="What does the EBA require regarding margin of conservatism in LGD estimation?",
    )
    col_ask, col_new = st.columns([1, 1])
    ask = col_ask.button("Ask", type="primary")
    if st.session_state["history"] and col_new.button("New conversation"):
        st.session_state["history"] = []
        st.session_state.pop("answer", None)
        st.rerun()

    if st.session_state["history"]:
        with st.expander(f"Conversation so far ({len(st.session_state['history']) // 2} turns)"):
            for turn in st.session_state["history"]:
                who = "**You:** " if turn["role"] == "user" else "**Copilot:** "
                st.markdown(who + turn["content"])

    render_sidebar()

    if ask and question.strip():
        _run_streaming_ask(question.strip(), doc_ids)
    elif st.session_state.get("answer"):
        render_answer(st.session_state["answer"])


if __name__ == "__main__":
    # `streamlit run app/ui.py` executes this module as __main__.
    main()
