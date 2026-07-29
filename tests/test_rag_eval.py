"""RAG evaluation tests (SPEC §11.3): judge parsing + config aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import evaluation.eval_rag as eval_rag
from app.rag import Answer, Citation, SourceChunk
from evaluation.judge import JudgeResult, parse_judge


# --- judge parsing ---------------------------------------------------------- #
def test_parse_judge_valid_json() -> None:
    text = '{"relevance": "RELEVANT", "citations_supported": true, "reason": "ok"}'
    result = parse_judge(text)
    assert result.relevance == "RELEVANT"
    assert result.citations_supported is True
    assert result.reason == "ok"


def test_parse_judge_extracts_embedded_json_and_normalizes_label() -> None:
    text = 'Sure!\n{"relevance": "partly_relevant", "citations_supported": false}'
    result = parse_judge(text)
    assert result.relevance == "PARTLY_RELEVANT"
    assert result.citations_supported is False


def test_parse_judge_defaults_on_garbage() -> None:
    result = parse_judge("no json here")
    assert result.relevance == "NON_RELEVANT"
    assert result.citations_supported is False


def test_parse_judge_unknown_label_falls_back() -> None:
    result = parse_judge('{"relevance": "MAYBE", "citations_supported": true}')
    assert result.relevance == "NON_RELEVANT"


# --- config aggregation (rag_answer + judge mocked) ------------------------- #
def _answer() -> Answer:
    return Answer(
        text="A [Doc, para. 1].",
        citations=[Citation(text="Doc, para. 1", doc_title="Doc", paras="1")],
        chunks_used=[
            SourceChunk(
                chunk_id="c1", doc_id="d1", doc_title="Doc", section_path=[],
                para_ids=["1"], pages=[1], score=1.0, text="src",
            )
        ],
        question="q", rewritten_query="q", retrieval_mode="bm25", prompt_version="v2",
        model="gpt-4o-mini", tokens_in=10, tokens_out=5, cost_usd=0.0002, latency_ms=300,
    )


def test_evaluate_config_aggregates(monkeypatch) -> None:
    gt = [{"question": "q1", "chunk_id": "c1"}, {"question": "q2", "chunk_id": "c1"}]
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    verdicts = iter(
        [
            JudgeResult("RELEVANT", True, "", 0.0001),
            JudgeResult("PARTLY_RELEVANT", False, "", 0.0001),
        ]
    )
    monkeypatch.setattr(eval_rag, "judge_answer", lambda *a, **k: next(verdicts))

    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-4o", log_to_db=False,
    )
    assert row["n"] == 2
    assert row["relevant"] == 1 and row["partly_relevant"] == 1
    assert row["relevant_rate"] == 0.5
    assert row["citation_supported_rate"] == 0.5
    assert row["avg_latency_ms"] == 300.0
    assert row["self_judged"] is False  # answer gpt-4o-mini vs judge gpt-4o


def test_evaluate_config_flags_self_judging(monkeypatch) -> None:
    gt = [{"question": "q1", "chunk_id": "c1"}]
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult("RELEVANT", True, "", 0.0)
    )
    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o", prompt_version="v2", judge_model="gpt-4o", log_to_db=False,
    )
    assert row["self_judged"] is True  # judge == answer model


def _config_row(model: str, prompt_version: str) -> dict:
    return {
        "model": model, "prompt_version": prompt_version, "n": 1, "self_judged": False,
        "relevant": 1, "partly_relevant": 0, "non_relevant": 0, "relevant_rate": 1.0,
        "citation_supported_rate": 1.0, "avg_answer_cost_usd": 0.0,
        "avg_judge_cost_usd": 0.0, "avg_latency_ms": 1.0,
    }


class _StubRetriever:
    def warm(self, **_kw) -> None:  # noqa: ANN003
        pass


def test_run_skips_failed_config_and_keeps_others(monkeypatch) -> None:
    # A config that errors (e.g. a sustained rate limit past retries) must not
    # discard the configs that already completed.
    settings = eval_rag.get_settings().model_copy(update={"eval_models": "m1,m2"})
    monkeypatch.setattr(eval_rag, "load_chunks", lambda kind, s=None: [])
    monkeypatch.setattr(eval_rag, "get_retriever", lambda: _StubRetriever())

    attempted: list[tuple[str, str]] = []

    def fake_eval(gt, ref, s, *, model, prompt_version, judge_model, log_to_db, workers=8):  # noqa: ANN001, ANN202
        attempted.append((model, prompt_version))
        if (model, prompt_version) == ("m1", "v2"):
            raise RuntimeError("simulated 429 after retries")
        return _config_row(model, prompt_version)

    monkeypatch.setattr(eval_rag, "evaluate_config", fake_eval)
    results = eval_rag.run(
        [{"question": "q", "chunk_id": "c"}], settings, judge_model="gpt-4o", log_to_db=False
    )
    assert len(attempted) == 4  # all 4 configs attempted (2 models x 2 prompts)
    assert len(results) == 3  # the failing one skipped, the rest kept
    assert ("m1", "v2") not in [(r["model"], r["prompt_version"]) for r in results]


# --- grafana provisioning sanity -------------------------------------------- #
def test_grafana_dashboard_has_five_plus_panels() -> None:
    root = Path(__file__).resolve().parent.parent
    dash = json.loads(
        (root / "monitoring/grafana/provisioning/dashboards/irb_copilot.json").read_text()
    )
    assert len(dash["panels"]) >= 5
    # Every panel targets the provisioned datasource with SQL.
    for panel in dash["panels"]:
        assert panel["targets"][0]["rawSql"].strip()
        assert panel["datasource"]["uid"] == "irb_postgres"
