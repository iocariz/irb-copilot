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


def render_details(ans: dict) -> None:
    """Citations, grounding warning, source snippets, and feedback buttons."""
    if ans.get("citations"):
        st.caption("Citations: " + " · ".join(c["text"] for c in ans["citations"]))
    if ans.get("ungrounded_citations"):
        st.warning(
            "⚠ These citations were not found in the retrieved sources "
            "(possible hallucination): " + "; ".join(ans["ungrounded_citations"])
        )
    st.markdown("#### Sources")
    for src in ans["chunks_used"]:
        header = f"{src['doc_title']} — para. {', '.join(src['para_ids'])} (p. {src['pages']})"
        with st.expander(header):
            if src["section_path"]:
                st.caption(" › ".join(src["section_path"]))
            st.write(src["text"])

    col_up, col_down, _ = st.columns([1, 1, 6])
    if col_up.button("👍", key=f"up_{ans.get('answer_id')}"):
        _submit_feedback(ans, "up")
    if col_down.button("👎", key=f"down_{ans.get('answer_id')}"):
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
        st.success("Thanks for the feedback!")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not send feedback: {exc}")


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
