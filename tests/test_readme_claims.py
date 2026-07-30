"""The README's evaluation numbers must match the committed CSVs.

The rubric asks for results copied from `evaluation/results/`, and prose drifts
from data silently: the README once cited a winner and four hit-rates that no
committed file supported, because the CSVs had been regenerated and the tables
had not. These tests read both artifacts and compare, so the next regeneration
fails loudly here instead of shipping stale claims.

They assert only numbers the README states as fact. Narrative is not pinned.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "evaluation" / "results"


def _hit_rates(csv_name: str) -> dict[tuple[str, str, str], float]:
    with (RESULTS / csv_name).open(encoding="utf-8") as fh:
        return {
            (r["retrieval_mode"], r["chunker"], r["rewrite_mode"]): float(r["hit_rate_at_5"])
            for r in csv.DictReader(fh)
        }


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def standard() -> dict[tuple[str, str, str], float]:
    return _hit_rates("retrieval_eval.csv")


@pytest.fixture(scope="module")
def debiased() -> dict[tuple[str, str, str], float]:
    return _hit_rates("retrieval_eval_hard.csv")


# --- the headline comparison table ------------------------------------------- #
@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid", "hybrid_rerank"])
def test_readme_table_matches_both_csvs(readme, standard, debiased, mode) -> None:
    """Each row is `| mode | <standard> | <de-biased> | <change> |`."""
    row = re.search(
        rf"^\|\s*\**{re.escape(mode)}\**\s*\|([^|]+)\|([^|]+)\|([^|]+)\|\s*$",
        readme,
        re.MULTILINE,
    )
    assert row, f"no README table row for {mode}"

    def number(cell: str) -> float:
        found = re.search(r"\d\.\d+", cell)
        assert found, f"no number in {cell!r} for {mode}"
        return float(found.group(0))

    expected_std = standard[(mode, "structure", "off")]
    expected_deb = debiased[(mode, "structure", "off")]
    assert number(row.group(1)) == pytest.approx(expected_std, abs=0.001)
    assert number(row.group(2)) == pytest.approx(expected_deb, abs=0.001)
    # The "change" column must be the actual difference, sign included.
    assert number(row.group(3)) == pytest.approx(abs(expected_deb - expected_std), abs=0.002)
    assert "−" in row.group(3) or "-" in row.group(3), "change column should be negative"


def test_readme_quotes_the_real_debiased_winner(readme, debiased) -> None:
    best = max(debiased.items(), key=lambda kv: kv[1])
    (mode, chunker, rewrite), hit = best
    assert f"`{mode}` + `{chunker}` + `{rewrite}`" in readme, "winner config not stated"
    assert f"{hit:.4f}" in readme, f"winning hit@5 {hit:.4f} not quoted"


def test_the_default_wins_both_ground_truths(readme, standard, debiased) -> None:
    """The README's central claim about the default: it wins on *both* sets, so the
    choice does not rest on trusting the de-biased set. If a future run breaks
    that, the argument has to change — not just the numbers."""
    std_best = max(standard.items(), key=lambda kv: kv[1])[0]
    deb_best = max(debiased.items(), key=lambda kv: kv[1])[0]
    assert std_best == deb_best, (
        f"README claims one config wins both, but standard={std_best} "
        f"and de-biased={deb_best}"
    )
    assert "wins **both** ground" in readme
    # Both winning scores must be quoted.
    assert f"**{standard[std_best]:.3f}**" in readme
    assert f"**{debiased[deb_best]:.4f}**" in readme


def test_readme_states_bm25_rank_collapse(readme, standard, debiased) -> None:
    """BM25 falling from 2nd to last is the bias evidence; verify it is true."""
    order = lambda d: sorted(  # noqa: E731
        ["bm25", "vector", "hybrid", "hybrid_rerank"],
        key=lambda m: -d[(m, "structure", "off")],
    )
    assert order(standard).index("bm25") == 1
    assert order(debiased).index("bm25") == 3
    assert "falls from second to last" in readme.lower()


def test_readme_states_the_rank_reversal_correctly(readme, standard, debiased) -> None:
    for label, data in (("standard", standard), ("de-biased", debiased)):
        order = sorted(
            ["bm25", "vector", "hybrid", "hybrid_rerank"],
            key=lambda m: -data[(m, "structure", "off")],
        )
        expected = " > ".join(f"`{m}`" if False else m for m in order)
        assert expected in readme, f"{label} rank order {expected!r} not in README"


# --- claims that must stay true of the data --------------------------------- #
def test_config_count_claim_matches_the_csvs(readme, standard, debiased) -> None:
    assert len(standard) == len(debiased) == 24
    assert "24 configs" in readme


_MODES = ["bm25", "vector", "hybrid", "hybrid_rerank"]
_CHUNKERS = ["structure", "naive"]


def _count(data, better: str, worse: str) -> int:
    return sum(
        1
        for mode in _MODES
        for chunker in _CHUNKERS
        if data[(mode, chunker, better)] >= data[(mode, chunker, worse)]
    )


def test_llm_rewrite_is_harmful_on_both_sets(readme, standard, debiased) -> None:
    """The strong half of the claim: the LLM reformulation loses 8/8 everywhere.
    This is what justifies shipping REWRITE_MODE=off."""
    assert _count(standard, "glossary", "llm") == 8
    assert _count(debiased, "glossary", "llm") == 8
    assert "8/8" in readme


def test_glossary_claim_is_stated_only_where_it_is_measurable(readme) -> None:
    """The weak half: glossary expansion is consistently negative on the standard
    set but a tie on the de-biased one, where only 3% of questions contain an
    acronym. The README must not over-claim a monotone 8/8 on both sets."""
    assert _count(standard_data(), "off", "glossary") == 8
    debiased_holds = _count(debiased_data(), "off", "glossary")
    assert debiased_holds < 8, "if this now holds 8/8, the README understates it"
    assert "6/8" in readme  # the honest de-biased count
    assert "indistinguishable from zero" in readme


def standard_data() -> dict[tuple[str, str, str], float]:
    return _hit_rates("retrieval_eval.csv")


def debiased_data() -> dict[tuple[str, str, str], float]:
    return _hit_rates("retrieval_eval_hard.csv")


def test_structure_beats_naive_claim_holds(readme, standard, debiased) -> None:
    """README asserts structure > naive in 12/12 comparisons, both sets."""
    for data in (standard, debiased):
        holds = sum(
            1
            for mode in ["bm25", "vector", "hybrid", "hybrid_rerank"]
            for rewrite in ["off", "glossary", "llm"]
            if data[(mode, "structure", rewrite)] > data[(mode, "naive", rewrite)]
        )
        assert holds == 12
    assert "12/12" in readme  # emphasis may vary; the claim must be present


def test_rerank_rescues_naive_chunking_numbers(readme, standard) -> None:
    hybrid = standard[("hybrid", "naive", "off")]
    rerank = standard[("hybrid_rerank", "naive", "off")]
    assert rerank > hybrid
    assert f"{hybrid:.3f}" in readme and f"{rerank:.3f}" in readme


# --- the RAG table --------------------------------------------------------- #
@pytest.fixture(scope="module")
def rag_rows() -> list[dict]:
    with (RESULTS / "rag_eval.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize(
    ("model", "prompt"),
    [("gpt-4o-mini", "v1"), ("gpt-4o-mini", "v2"), ("gpt-4o", "v1"), ("gpt-4o", "v2")],
)
def test_readme_rag_row_matches_the_csv(readme, rag_rows, model, prompt) -> None:
    record = next(
        r for r in rag_rows if r["model"] == model and r["prompt_version"] == prompt
    )
    row = re.search(
        rf"^\|\s*\**{re.escape(model)}\**\s*\|\s*\**{prompt}\**\s*\|([^|]+)\|([^|]+)\|",
        readme,
        re.MULTILINE,
    )
    assert row, f"no README RAG row for {model}/{prompt}"

    def number(cell: str) -> float:
        found = re.search(r"\d\.\d+", cell)
        assert found, f"no number in {cell!r}"
        return float(found.group(0))

    assert number(row.group(1)) == pytest.approx(float(record["relevant_rate"]), abs=0.005)
    assert number(row.group(2)) == pytest.approx(
        float(record["citation_supported_rate"]), abs=0.005
    )


def test_readme_names_the_highest_relevance_config_even_when_not_the_default(
    readme, rag_rows
) -> None:
    """The chosen default is not the top-scoring config — it trades 4pt of
    relevance for 15x lower cost. The README must say so rather than let a reader
    assume the default won, so this pins the honest framing."""
    complete = [r for r in rag_rows if r["answered"] == r["n"]]
    best = max(complete, key=lambda r: float(r["relevant_rate"]))
    default = next(
        r for r in complete if r["model"] == "gpt-4o-mini" and r["prompt_version"] == "v2"
    )
    if (best["model"], best["prompt_version"]) != ("gpt-4o-mini", "v2"):
        assert "not* the\nhighest-relevance config" in readme or (
            "not* the" in readme and "highest-relevance config" in readme
        ), "README must disclose that the default is not the top scorer"
        assert f"`{best['model']}` + `{best['prompt_version']}`" in readme
        # And the trade must be quantified, not hand-waved.
        gap = round(float(best["relevant_rate"]) - float(default["relevant_rate"]), 2)
        assert f"{int(gap * 100)}pt of relevance" in readme


def test_readme_reports_tied_citation_support(readme, rag_rows) -> None:
    """The default's case rests on citation support being tied, so verify it."""
    rates = {
        (r["model"], r["prompt_version"]): float(r["citation_supported_rate"])
        for r in rag_rows
    }
    default = rates[("gpt-4o-mini", "v2")]
    best_relevance = max(rag_rows, key=lambda r: float(r["relevant_rate"]))
    assert default == rates[(best_relevance["model"], best_relevance["prompt_version"])]
    assert "identical** citation support" in readme


def test_readme_does_not_publish_contaminated_latency(readme) -> None:
    """A 606s outlier moved a mean from ~3s to ~18s. The README must explain the
    omission rather than quote the mean as model latency."""
    section = readme.split("### RAG (`eval_rag.py`)")[1]
    assert "Latency is deliberately not reported" in section
    assert "| cost/q |" in section and "| latency |" not in section


def test_readme_rag_completeness_claim_is_true(readme, rag_rows) -> None:
    """The README says every config answered and judged all 100 with zero parse
    errors — that must actually hold in the artifact."""
    assert len(rag_rows) == 4
    for r in rag_rows:
        assert r["answered"] == r["n"] == "100"
        assert r["judged"] == "100"
        assert r["parse_errors"] == "0"
        assert r["answer_errors"] == "0"
    assert "zero parse errors" in readme


def test_readme_does_not_reintroduce_unbacked_rag_numbers(readme, rag_rows) -> None:
    """Guards the original defect: a table of rates no committed file supported."""
    section = readme.split("### RAG (`eval_rag.py`)")[1]
    quoted = {float(m) for m in re.findall(r"\| (0\.\d\d) \|", section)}
    real = {round(float(r[k]), 2) for r in rag_rows
            for k in ("relevant_rate", "citation_supported_rate")}
    assert quoted <= real, f"README quotes rates absent from the CSV: {quoted - real}"


def test_readme_names_the_configured_judge_model(readme) -> None:
    from app.config import get_settings

    assert f"`{get_settings().judge_model}`" in readme


def test_readme_documents_the_shipped_rewrite_mode(readme) -> None:
    from app.config import get_settings

    assert f"`REWRITE_MODE={get_settings().rewrite_mode}`" in readme


def test_readme_env_table_uses_current_variable_names(readme) -> None:
    # ENABLE_REWRITE is a deprecated migration alias, not live configuration.
    assert "| `REWRITE_MODE` |" in readme
    assert "| `ENABLE_REWRITE` |" not in readme
