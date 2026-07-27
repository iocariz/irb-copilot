"""Streamlit front end (SPEC §10).

A single-page app: ask a question (optionally filtered to specific documents),
see the cited answer with expandable source snippets, rate it 👍/👎, and view the
model/cost/latency of the last answer in the sidebar. Calls the FastAPI backend;
holds no chat history (single-turn).
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/ui.py` puts app/ (not the repo root) on sys.path, so
# `import app.*` would fail; add the repo root before importing app modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import streamlit as st
import yaml

from app.config import get_settings

settings = get_settings()
_TIMEOUT = httpx.Timeout(120.0)


def load_documents() -> dict[str, str]:
    """Return {doc_id: title} from data/sources.yaml for the document filter."""
    path = settings.data_path / "sources.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {d["id"]: d["title"] for d in raw.get("documents", [])}


def ask_api(question: str, doc_ids: list[str] | None) -> dict:
    resp = httpx.post(
        f"{settings.api_url}/ask",
        json={"question": question, "doc_ids": doc_ids or None},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_feedback(answer_id: str, thumbs: str, comment: str | None = None) -> None:
    httpx.post(
        f"{settings.api_url}/feedback",
        json={"answer_id": answer_id, "thumbs": thumbs, "comment": comment},
        timeout=_TIMEOUT,
    ).raise_for_status()


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
    if ans.get("rewritten_query") and ans["rewritten_query"] != ans["question"]:
        st.sidebar.caption(f"rewritten: {ans['rewritten_query']}")


def render_answer(ans: dict) -> None:
    st.markdown("### Answer")
    st.write(ans["text"])

    if ans.get("citations"):
        st.caption("Citations: " + " · ".join(c["text"] for c in ans["citations"]))

    st.markdown("### Sources")
    for src in ans["chunks_used"]:
        header = f"{src['doc_title']} — para. {', '.join(src['para_ids'])} (p. {src['pages']})"
        with st.expander(header):
            if src["section_path"]:
                st.caption(" › ".join(src["section_path"]))
            st.write(src["text"])

    col_up, col_down, _ = st.columns([1, 1, 6])
    if col_up.button("👍", key="up"):
        _submit_feedback(ans, "up")
    if col_down.button("👎", key="down"):
        _submit_feedback(ans, "down")


def _submit_feedback(ans: dict, thumbs: str) -> None:
    if not ans.get("answer_id"):
        st.warning("Answer was not logged; feedback unavailable.")
        return
    try:
        send_feedback(ans["answer_id"], thumbs)
        st.success("Thanks for the feedback!")
    except Exception as exc:  # noqa: BLE001 — surface a friendly message.
        st.error(f"Could not send feedback: {exc}")


def main() -> None:
    st.set_page_config(page_title="IRB Copilot", page_icon="📘")
    st.title("📘 IRB Copilot")
    st.caption(
        "Q&A over EU prudential regulation for credit-risk modeling. "
        "Answers cite their sources."
    )

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

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and answering…"):
            try:
                st.session_state["answer"] = ask_api(question.strip(), doc_ids)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")

    render_sidebar()
    if st.session_state.get("answer"):
        render_answer(st.session_state["answer"])


if __name__ == "__main__":
    # `streamlit run app/ui.py` executes this module as __main__.
    main()
