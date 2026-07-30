"""RAG evaluation tests (SPEC §11.3): judge parsing + config aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import evaluation.eval_rag as eval_rag
from app.rag import Answer, Citation, SourceChunk
from evaluation.judge import PARSE_ERROR, RELEVANCE_LABELS, JudgeResult, parse_judge


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


def test_parse_judge_reports_garbage_as_parse_error_not_a_verdict() -> None:
    """An unreadable response is missing data. Scoring it NON_RELEVANT (as this
    used to) silently pushes every config's score down."""
    result = parse_judge("no json here")
    assert result.relevance == PARSE_ERROR
    assert result.parsed is False
    assert result.citations_supported is False
    assert "no json here" in result.reason  # excerpt kept for diagnosis


def test_parse_judge_treats_an_unknown_label_as_a_parse_error() -> None:
    result = parse_judge('{"relevance": "MAYBE", "citations_supported": true}')
    assert result.relevance == PARSE_ERROR
    assert result.parsed is False


def test_parse_judge_marks_real_verdicts_as_parsed() -> None:
    for label in RELEVANCE_LABELS:
        result = parse_judge(f'{{"relevance": "{label}", "citations_supported": true}}')
        assert result.relevance == label and result.parsed is True


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
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False,
    )
    assert row["n"] == 2
    assert row["relevant"] == 1 and row["partly_relevant"] == 1
    assert row["relevant_rate"] == 0.5
    assert row["citation_supported_rate"] == 0.5
    assert row["avg_latency_ms"] == 300.0
    # gpt-4o-mini answers judged by gpt-5.4-mini: different families, unbiased.
    assert row["self_judged"] is False


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
        "answered": 1, "answer_errors": 0, "judged": 1, "parse_errors": 0,
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


# --- self-preference detection (judge vs answer model family) --------------- #
def test_model_family_strips_the_size_suffix() -> None:
    assert eval_rag.model_family("gpt-4o-mini") == "gpt-4o"
    assert eval_rag.model_family("gpt-4o") == "gpt-4o"
    assert eval_rag.model_family("gpt-5.4-mini") == "gpt-5.4"
    assert eval_rag.model_family("gpt-5-mini") == "gpt-5"


def test_self_judged_is_family_based_not_name_equality() -> None:
    # Exact-name equality is too narrow: a gpt-5.4 judge scoring gpt-5.4-mini
    # answers is the same self-preference failure mode.
    assert eval_rag.is_self_judged("gpt-5.4-mini", "gpt-5.4") is True
    assert eval_rag.is_self_judged("gpt-4o-mini", "gpt-4o") is True
    assert eval_rag.is_self_judged("gpt-4o-mini", "gpt-4o-mini") is True
    # The configured default pairing must be cross-family.
    assert eval_rag.is_self_judged("gpt-4o-mini", "gpt-5.4-mini") is False


def test_default_judge_is_cross_family_with_the_default_answer_model() -> None:
    from app.config import get_settings

    settings = get_settings()
    assert not eval_rag.is_self_judged(settings.llm_model, settings.judge_model)


# --- parse errors leave the denominator, they are not negative verdicts (T3) - #
def test_parse_errors_are_excluded_from_the_rates(monkeypatch) -> None:
    """Two RELEVANT and two unreadable verdicts is a 1.0 relevant-rate over the
    two answers actually judged — not 0.5 over four."""
    gt = [{"question": f"q{i}", "chunk_id": "c1"} for i in range(4)]
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    verdicts = iter([
        JudgeResult("RELEVANT", True, "", 0.0),
        JudgeResult(PARSE_ERROR, False, "garbage", 0.0),
        JudgeResult("RELEVANT", True, "", 0.0),
        JudgeResult(PARSE_ERROR, False, "garbage", 0.0),
    ])
    monkeypatch.setattr(eval_rag, "judge_answer", lambda *a, **k: next(verdicts))

    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False,
    )
    assert row["n"] == 4 and row["judged"] == 2 and row["parse_errors"] == 2
    assert row["relevant_rate"] == 1.0  # 2/2, NOT 2/4
    assert row["citation_supported_rate"] == 1.0
    # A parse error must not be counted as a NON_RELEVANT answer.
    assert row["non_relevant"] == 0


def test_all_parse_errors_yields_zero_rates_not_a_crash(monkeypatch) -> None:
    gt = [{"question": "q", "chunk_id": "c1"}]
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult(PARSE_ERROR, False, "x", 0.0)
    )
    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False,
    )
    assert row["judged"] == 0 and row["relevant_rate"] == 0.0


def test_unreadable_verdicts_are_logged_as_null_relevance(monkeypatch) -> None:
    """The Grafana relevance panel should show real judgements only."""
    logged: list[object] = []
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult(PARSE_ERROR, False, "x", 0.0)
    )
    monkeypatch.setattr(
        eval_rag,
        "log_conversation",
        lambda ans, judge_relevance=None, source=None: logged.append(
            (judge_relevance, source)
        )
        or "id",
    )
    eval_rag.evaluate_config(
        [{"question": "q", "chunk_id": "c1"}], {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=True,
    )
    from monitoring.db import EVAL

    # NULL relevance (no readable verdict) AND tagged as evaluation traffic.
    assert logged == [(None, EVAL)]


def test_judge_retries_once_then_accepts_a_readable_verdict(monkeypatch) -> None:
    import evaluation.judge as judge_mod
    from app.providers import LLMResult

    replies = iter([
        LLMResult("truncated {", "m", 10, 5, 0.001, 1, finish_reason="length"),
        LLMResult('{"relevance": "RELEVANT", "citations_supported": true}', "m", 10, 5, 0.002, 1),
    ])
    monkeypatch.setattr(judge_mod, "complete", lambda *a, **k: next(replies))
    result = judge_mod.judge_answer("q", "ref", "ans", "src", model="m")
    assert result.relevance == "RELEVANT" and result.parsed is True
    assert result.cost_usd == 0.003  # every attempt is billed and accounted for


def test_judge_gives_up_as_parse_error_after_the_retry(monkeypatch) -> None:
    import evaluation.judge as judge_mod
    from app.providers import LLMResult

    monkeypatch.setattr(
        judge_mod, "complete",
        lambda *a, **k: LLMResult("nope", "m", 10, 5, 0.001, 1, finish_reason="length"),
    )
    result = judge_mod.judge_answer("q", "ref", "ans", "src", model="m")
    assert result.relevance == PARSE_ERROR and result.parsed is False
    assert result.cost_usd == 0.002  # both attempts billed


# --- rate-limit pacing + per-question resilience ---------------------------- #
def test_low_tpm_models_are_paced_to_one_request() -> None:
    """gpt-4o at 30k TPM sustains ~8 req/min; 3 concurrent workers attempted ~45
    and every retry's freed budget was eaten by a sibling, so the SDK's backoff
    never converged. Serialising it lets backoff work."""
    from app.providers import max_concurrency

    assert max_concurrency("gpt-4o", 8) == 1
    assert max_concurrency("gpt-4o", 1) == 1
    # Unconstrained models keep the requested concurrency.
    assert max_concurrency("gpt-4o-mini", 8) == 8
    assert max_concurrency("gpt-5.4-mini", 3) == 3
    # Never returns 0, whatever is requested.
    assert max_concurrency("gpt-4o", 0) == 1


def test_one_failing_question_does_not_sink_the_config(monkeypatch) -> None:
    """A rate limit outlasting retries used to raise out of pool.map and discard
    the whole config — 99 good answers lost to one failure."""
    gt = [{"question": f"q{i}", "chunk_id": "c1"} for i in range(4)]
    calls = {"n": 0}

    def flaky(_q, settings=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("429 rate limit, retries exhausted")
        return _answer()

    monkeypatch.setattr(eval_rag, "rag_answer", flaky)
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult("RELEVANT", True, "", 0.0)
    )
    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False, workers=1,
    )
    assert row["n"] == 4
    assert row["answer_errors"] == 1 and row["answered"] == 3
    assert row["judged"] == 3
    assert row["relevant_rate"] == 1.0  # 3/3, the failure is excluded not scored
    assert row["non_relevant"] == 0  # never counted as a bad answer


def test_cost_and_latency_average_over_answered_not_attempted(monkeypatch) -> None:
    gt = [{"question": f"q{i}", "chunk_id": "c1"} for i in range(2)]
    calls = {"n": 0}

    def half_fail(_q, settings=None):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return _answer()

    monkeypatch.setattr(eval_rag, "rag_answer", half_fail)
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult("RELEVANT", True, "", 0.0)
    )
    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False, workers=1,
    )
    # One answer at $0.0002 / 300ms: averaging over 2 attempts would halve both.
    assert row["avg_answer_cost_usd"] == pytest.approx(0.0002)
    assert row["avg_latency_ms"] == pytest.approx(300.0)


# --- latency percentiles, not just the mean --------------------------------- #
def test_percentiles_resist_a_single_stalled_request() -> None:
    """One 10-minute stall moved a real config's mean from ~3s to ~18s, making a
    cheap model look 7x slower than an expensive one. The median must not move."""
    from evaluation.eval_rag import _percentile

    normal = [3000.0] * 99
    assert _percentile(normal + [600_000.0], 0.50) == 3000.0
    assert _percentile([], 0.50) == 0.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_config_row_reports_latency_percentiles(monkeypatch) -> None:
    gt = [{"question": f"q{i}", "chunk_id": "c1"} for i in range(3)]
    monkeypatch.setattr(eval_rag, "rag_answer", lambda q, settings=None: _answer())
    monkeypatch.setattr(
        eval_rag, "judge_answer", lambda *a, **k: JudgeResult("RELEVANT", True, "", 0.0)
    )
    row = eval_rag.evaluate_config(
        gt, {"c1": "ref"}, eval_rag.get_settings(),
        model="gpt-4o-mini", prompt_version="v2", judge_model="gpt-5.4-mini",
        log_to_db=False, workers=1,
    )
    assert row["p50_latency_ms"] == 300.0 and row["p95_latency_ms"] == 300.0
