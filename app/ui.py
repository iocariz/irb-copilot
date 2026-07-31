"""Streamlit front end (SPEC §10).

Ask a question, watch the cited answer stream in, and check the sources behind
it. Sources the answer actually cited are marked and opened first, because
verifying a claim against the regulation is the task this tool exists for; the
rest stay collapsed. Follow-ups keep a short conversation history, and 👍/👎
feeds the monitoring dashboard. Calls the FastAPI backend's streaming endpoint.

Controls and telemetry live in the sidebar so the answer starts at the top of
the page. Presentation is set by `.streamlit/config.toml` (palette, so Streamlit's
own widgets inherit it) plus the `_CSS` sheet below (typography and source
cards, which have no theme hooks).
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

# Presentation. The palette itself lives in .streamlit/config.toml so Streamlit's
# own widgets pick it up; this sheet handles typography and the source cards,
# which have no theme hooks.
#
# Design intent: an instrument for reading regulation, not a dashboard. Answer
# and source text are set in a serif at a readable measure because that is what
# the user is here to read; everything else (labels, metadata, controls) is small
# sans, deliberately quiet. Navy marks a source the answer actually cited —
# amber and red are left unused by the chrome so a warning still carries weight.
#
# Selectors are limited to `data-testid` hooks and classes emitted below.
# Streamlit's generated class names change between releases and are not touched.
_CSS = """
<style>
  :root {
    --ink: #1B1B19;
    --ink-soft: #5C594F;
    --rule: #DCD8CE;
    --accent: #234E70;
    --accent-wash: #EDF1F5;
    --serif: Charter, "Iowan Old Style", Georgia, "Times New Roman", serif;
  }

  /* Reading measure: long lines of regulation are hard to track. */
  .block-container { max-width: 62rem; padding-top: 2.2rem; }

  /* Masthead */
  .irb-masthead { border-bottom: 2px solid var(--ink); margin-bottom: 1.6rem; }
  .irb-masthead h1 {
    font-family: var(--serif); font-size: 2.05rem; font-weight: 600;
    letter-spacing: -0.015em; margin: 0 0 .15rem 0; color: var(--ink);
  }
  .irb-masthead p {
    font-size: .82rem; color: var(--ink-soft); margin: 0 0 .7rem 0;
    letter-spacing: .01em;
  }

  /* Small caps metadata labels, used for section headings and source meta. */
  .irb-label {
    font-size: .68rem; text-transform: uppercase; letter-spacing: .11em;
    color: var(--ink-soft); font-weight: 600; margin: 1.4rem 0 .45rem 0;
  }

  /* Answer body: this is the thing being read. */
  .irb-answer, .irb-answer p, .irb-answer li {
    font-family: var(--serif); font-size: 1.06rem; line-height: 1.62;
    color: var(--ink);
  }

  /* Citations rendered inline by the model, e.g. [Doc, para. 82]. */
  .irb-cites code, .irb-chip {
    display: inline-block; font-size: .72rem; padding: .12rem .42rem;
    border: 1px solid var(--rule); border-radius: 3px; background: #fff;
    color: var(--ink-soft); margin: 0 .25rem .25rem 0;
  }

  /* Source cards. A left rule carries the signal: navy = the answer cited it. */
  div[data-testid="stExpander"] {
    border: 1px solid var(--rule) !important; border-radius: 0 !important;
    border-left: 3px solid var(--rule) !important; margin-bottom: .4rem;
    background: #fff;
  }
  div[data-testid="stExpander"]:has(.irb-cited) {
    border-left-color: var(--accent) !important; background: var(--accent-wash);
  }
  div[data-testid="stExpander"] summary { font-size: .86rem; }
  div[data-testid="stExpander"] p { font-family: var(--serif); line-height: 1.55; }

  .irb-meta {
    font-size: .7rem; color: var(--ink-soft); letter-spacing: .02em;
    border-top: 1px dotted var(--rule); padding-top: .35rem; margin-top: .5rem;
  }
  .irb-rank {
    font-variant-numeric: tabular-nums; font-weight: 700; color: var(--accent);
  }

  /* Sidebar reads as an instrument panel, not a second page. */
  section[data-testid="stSidebar"] { border-right: 1px solid var(--rule); }
  section[data-testid="stSidebar"] .irb-stat {
    display: flex; justify-content: space-between; gap: 1rem;
    font-size: .78rem; padding: .28rem 0; border-bottom: 1px dotted var(--rule);
  }
  section[data-testid="stSidebar"] .irb-stat span:last-child {
    font-variant-numeric: tabular-nums; color: var(--ink);
  }
  section[data-testid="stSidebar"] .irb-stat span:first-child { color: var(--ink-soft); }

  /* Alerts: keep their semantics, lose the sticker look. */
  div[data-testid="stAlert"] { border-radius: 2px; font-size: .86rem; }
</style>
"""


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
        chips = "".join(
            f'<span class="irb-chip">{c["text"]}</span>' for c in ans["citations"]
        )
        st.markdown(f'<div class="irb-cites">{chips}</div>', unsafe_allow_html=True)
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


def _label(text: str) -> None:
    """A small-caps section label."""
    st.markdown(f'<div class="irb-label">{text}</div>', unsafe_allow_html=True)


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
    _label(f"Sources · {len(cited)} of {len(sources)} cited")
    for rank, src in cited:
        with st.expander(f"#{rank}  {source_header(src)}", expanded=True):
            # Marker class the stylesheet keys the navy left rule off.
            st.markdown(
                '<div class="irb-cited"></div>Backs '
                + " ".join(f'<span class="irb-chip">{c}</span>' for c in src["cited_by"]),
                unsafe_allow_html=True,
            )
            _render_source_body(src, rank, len(sources), retrieval_mode)
    if uncited:
        st.caption(
            f"{len(uncited)} further source(s) were retrieved but not cited by the answer."
        )
        for rank, src in uncited:
            with st.expander(f"#{rank}  {source_header(src)}"):
                _render_source_body(src, rank, len(sources), retrieval_mode)


def _render_source_body(
    src: dict, rank: int, total: int, retrieval_mode: str | None
) -> None:
    st.write(src["text"])
    trail = " › ".join(src.get("section_path") or []) or "—"
    st.markdown(
        f'<div class="irb-meta"><span class="irb-rank">#{rank}</span> of {total} · '
        f'{format_score(src.get("score", 0.0), retrieval_mode)} · {trail}</div>',
        unsafe_allow_html=True,
    )


def _render_feedback(ans: dict) -> None:
    """Thumbs buttons, or a record of the rating already given.

    Each click used to POST a new `feedback` row, so a user clicking twice skewed
    the up/down ratio panel — which has the least data of any panel and so is the
    most distorted by duplicates.
    """
    answer_id = ans.get("answer_id")
    already = st.session_state.get(_feedback_state_key(answer_id)) if answer_id else None
    _label("Was this useful?")
    if already:
        st.caption(f"Recorded: {'👍' if already == 'up' else '👎'} — thank you.")
        return
    col_up, col_down, _ = st.columns([1, 1, 8])
    if col_up.button("👍", key=f"up_{answer_id}", use_container_width=True):
        _submit_feedback(ans, "up")
    if col_down.button("👎", key=f"down_{answer_id}", use_container_width=True):
        _submit_feedback(ans, "down")


def render_answer(ans: dict) -> None:
    """Static (non-streaming) render of a stored answer."""
    _label("Answer")
    st.markdown(f'<div class="irb-answer">{ans["text"]}</div>', unsafe_allow_html=True)
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


def render_sidebar(docs: dict[str, str]) -> list[str]:
    """Controls and last-answer telemetry. Returns the selected doc_ids.

    Filters live here rather than above the question so the answer starts at the
    top of the page — this is a reading tool, and the text is the point.
    """
    st.sidebar.markdown('<div class="irb-label">Corpus filter</div>', unsafe_allow_html=True)
    selected = st.sidebar.multiselect(
        "Restrict to documents", options=list(docs.values()),
        label_visibility="collapsed",
        placeholder=f"All {len(docs)} documents",
    )
    title_to_id = {title: doc_id for doc_id, title in docs.items()}

    if st.session_state.get("history"):
        turns = len(st.session_state["history"]) // 2
        st.sidebar.markdown(
            f'<div class="irb-label">Conversation · {turns} turn(s)</div>',
            unsafe_allow_html=True,
        )
        if st.sidebar.button("Start over", use_container_width=True):
            st.session_state["history"] = []
            st.session_state.pop("answer", None)
            st.rerun()

    ans = st.session_state.get("answer")
    st.sidebar.markdown('<div class="irb-label">Last answer</div>', unsafe_allow_html=True)
    if not ans:
        st.sidebar.caption("Ask a question to see model, cost and latency.")
    else:
        stats = [
            ("model", ans["model"]),
            ("cost", f"${ans['cost_usd']:.5f}"),
            ("latency", f"{ans['latency_ms']:,} ms"),
            ("tokens", f"{ans['tokens_in']:,} → {ans['tokens_out']:,}"),
            ("retrieval", ans["retrieval_mode"]),
            ("prompt", ans["prompt_version"]),
        ]
        st.sidebar.markdown(
            "".join(
                f'<div class="irb-stat"><span>{k}</span><span>{v}</span></div>'
                for k, v in stats
            ),
            unsafe_allow_html=True,
        )
    return [title_to_id[t] for t in selected]


def _run_streaming_ask(question: str, doc_ids: list[str]) -> None:
    history = st.session_state.get("history", [])
    _label("Answer")
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

    # Retrieval (embedding + cross-encoder) runs before the first token, so the
    # page would otherwise sit blank for ~half a second with no acknowledgement.
    slot = st.empty()
    with st.spinner("Searching the corpus…"), slot:
        streamed = st.write_stream(tokens())
    # st.write_stream renders raw, so a just-streamed answer would be sans-serif
    # while the same answer re-rendered after a rerun is serif. Swap in the styled
    # version once the text is complete, so the two paths look identical.
    if streamed:
        slot.markdown(
            f'<div class="irb-answer">{streamed}</div>', unsafe_allow_html=True
        )
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
    st.set_page_config(
        page_title="IRB Copilot",
        page_icon="§",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="irb-masthead">'
        "<h1>IRB Copilot</h1>"
        "<p>Question answering over EU prudential regulation for credit-risk "
        "modelling · every claim carries its source</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("history", [])

    doc_ids = render_sidebar(load_documents())

    question = st.text_area(
        "Your question",
        label_visibility="collapsed",
        height=88,
        placeholder=(
            "What does the EBA require regarding margin of conservatism in LGD "
            "estimation?"
        ),
    )
    asked = st.button("Ask", type="primary")
    if asked and not question.strip():
        # Used to do nothing at all, which reads as a broken button.
        st.warning("Enter a question first.")
        return

    if st.session_state["history"]:
        with st.expander("Conversation so far"):
            for turn in st.session_state["history"]:
                who = "**You**" if turn["role"] == "user" else "**Copilot**"
                st.markdown(f"{who} · {turn['content']}")

    if asked:
        _run_streaming_ask(question.strip(), doc_ids)
    elif st.session_state.get("answer"):
        render_answer(st.session_state["answer"])


if __name__ == "__main__":
    # `streamlit run app/ui.py` executes this module as __main__.
    main()
