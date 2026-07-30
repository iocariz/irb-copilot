# IRB Copilot — remediation tasks

Findings from the codebase review of 2026-07-29 (bugs, performance, methodology).
Baseline at review time: 161 tests pass, `ruff check` clean.

**Ordering matters.** Phase 1 changes what the evaluation *measures*, so the
retrieval CSVs must be regenerated afterwards — which is also what fixes the
README drift in Phase 2. Do not update the README tables before Phase 1 lands.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done

---

## Done — clean-clone check: `.env.example` shipped a broken API key (2026-07-30)

Cloned the pushed repo to a fresh directory and ran `make setup` + the suite as a
reviewer would. **17 API tests failed with 401** on the clean clone while passing
in the working copy.

Cause: `make setup` copies `.env.example` to `.env`, and

    API_KEY=                                    # if set, /ask ... require X-API-Key

parses as `api_key='# if set, /ask ... require X-API-Key'`. dotenv reads an inline
comment that follows an **empty** value as the value itself. (`RATE_LIMIT=30  #
...` is fine — the quirk only affects empty values.)

**Effect on a reviewer:** every `/ask` and `/feedback` returns 401 with an API key
nobody knows. The UI would look completely broken on first run. Invisible to us
because our own `.env` never set `API_KEY` at all — the failure existed only on
the path we never took.

- [x] Comment moved above the variable in `.env.example`; `API_KEY` now resolves
      to `''`
- [x] `test_env_example_has_no_value_that_is_actually_a_comment` scans every
      value in the example for the pattern, and a second test asserts settings
      built from it serve requests unauthenticated. Verified the guard fails on
      the original line
- [x] Clean clone re-verified: `make setup` from the fixed example, 305/305 green

**The general lesson:** the suite passed on every commit, because the suite ran
against a `.env` that had drifted from the file we ship. Config that only some
users receive needs testing against the artifact they actually get.

---

## Done — UI: source verification + two false-alarm classes (2026-07-30)

Started as UI polish; the second half turned out to be a correctness bug in the
hallucination check that T10 had introduced.

### Rendering bugs (one of them mine)

- [x] **Source headers were broken for 24% of chunks.** T10 correctly stopped
      inventing paragraph anchors for annexes and tables, but `ui.py` still
      rendered `— para.  (p. [1])` — an empty anchor and a raw Python list.
      Unnumbered chunks now fall back to their section (`§ Contents`), pages
      render as `p. 45` / `pp. 12–14`, and a section label that merely repeats
      the document title is skipped (3/535 chunks, but it looks broken).
      I introduced this in T10 and did not check the render path.
- [x] **Feedback could be submitted repeatedly** — every click POSTed a new row,
      skewing the up/down panel, which has the least data of any panel and so is
      the most distorted by duplicates. State is set only after a successful
      POST, so a network failure still allows a retry.
- [x] **An answer with no citations rendered silently.** That is the exact
      failure this project exists to prevent. Worded neutrally, because a correct
      refusal ("the context does not cover this") also has zero citations, and a
      warning that fires on correct behaviour gets ignored.

### The real find: 34% of hallucination flags were false

`prompts.citation_header` shows a chunk with no paragraph number as
`[Title, para. n/a]` — so that is the only way an answer *can* cite one. But
`check_citation_grounding` looked for "n/a" among the retrieved paragraph
numbers, found nothing, and flagged it. Since T10 made 24% of chunks unnumbered,
**every citation of an annex or table was reported as a possible hallucination**.

Measured on the last evaluation run: 278 of 1,497 answers flagged (19%), of which
**95 (34%) mention "n/a"**. A second class was visible in the same data —
`para. 91(d)` failed against a chunk carrying `91`, because sub-point suffixes
were compared literally.

- [x] `cited_paragraph_forms()` — a sub-point is satisfied by its parent
      paragraph; `n/a` is satisfied by an unnumbered chunk from that document
- [x] The all-paragraphs rule is preserved: `para. 91, 999` with only 91
      retrieved is still flagged. Tested explicitly, since the fix is a
      *loosening* and that is exactly what a loosening tends to break

A noisy hallucination warning is worse than none — users learn to dismiss it, in
the one domain where they most need to read it.

### Source verification (the point of the exercise)

- [x] `attribute_citations()` fills `SourceChunk.cited_by`, computed server-side
      with the **same matcher** as the grounding check, so the UI cannot mark a
      source as backing a citation the grounding check calls unsupported. Tested
      by asserting the two agree.
- [x] The UI lists cited sources first, marked `✓` and already expanded, showing
      which citation each one backs; uncited sources stay collapsed under a
      count. Verifying a claim no longer means opening every source in turn.

### Transparency (items 5-6)

- [x] **The query retrieval actually ran on** is shown when it differs from the
      question. Live example: "and what is the materiality threshold for that?"
      was retrieved as "What is the materiality threshold for a material
      obligation as defined in EU prudential regulation?" — a completely
      different query the user could not previously see. First thing worth
      checking when the sources look wrong.
- [x] **Retrieval rank shown explicitly**, *because* grouping cited sources first
      destroyed the rank signal that used to be implicit in the ordering. Fixing
      one thing removed information elsewhere; worth noticing before shipping.
- [x] **Raw score shown, always labelled with its mode.** Scores are not
      comparable across modes — a top `hybrid_rerank` hit reads ~0.4 where a
      `hybrid` one reads ~0.016 for the same quality of match — so a bare number
      would be read as a confidence it is not. Rank is the comparable signal.

Verified on real answers: attribution correct (3 of 5, then 2 of 5 cited),
condensed follow-up query surfaced, scores and ranks rendered.
`tests/test_ui.py` (19) + 6 attribution tests. Suite green (305).

---

## Done — T11 (+ T14): eval traffic separated from real usage (2026-07-30)

`eval_rag` seeds hundreds of judged conversations so the Grafana judge panel has
data, but nothing distinguished them from real `/ask` calls. Measured on the live
database: **1,901 of 1,921 rows (99%) were evaluation traffic.** Every usage,
cost and latency panel was describing the harness's own activity as though users
had produced it — a monitoring dashboard that was, in practice, monitoring itself.

- [x] `conversations.source` (`'live'` | `'eval'`), NOT NULL with a server
      default so provenance can never be unknown; constants `LIVE`/`EVAL` in
      `monitoring.db` because the panels filter on the literal values
- [x] `log_conversation(..., source=LIVE)` by default — a real `/ask` does not
      have to remember to say so; `eval_rag._safe_log` passes `EVAL`
- [x] Idempotent migration in `_ADDED_COLUMNS` + matching `schema.sql`, plus an
      `(source, ts)` index since every usage panel now filters on it
- [x] **Historical rows backfilled** by the one rule that identifies them
      exactly: `judge_relevance IS NOT NULL` is set only by the evaluation.
      1,901 rows reclassified; the 20 genuine ones remain `live`
- [x] Panels 1–5, 7, 8 filter `source = 'live'`; panel 6 (judge relevance) reads
      `source = 'eval'`. The feedback panel joins through to `conversations` so
      its scope is stated rather than assumed
- [x] New panel: **truncated-answer rate**

### T14 finished here, as planned — it needed this migration

- [x] `Answer.truncated`, set from `LLMResult.finish_reason == "length"`
- [x] Persisted as `conversations.answer_truncated`
- [x] The Streamlit UI warns when an answer was cut off. Previously it simply
      stopped mid-sentence and read as complete, which in a citation-critical
      domain is worse than an error

### Tests: `tests/test_monitoring.py` (16 new)

Structural rather than behavioural, because the failure mode is a *missing
filter*, which no unit test of application code would catch:

- every panel must state its scope — `test_every_panel_states_its_scope` fails if
  any new panel forgets, which is exactly how this bug would return
- each usage panel individually asserted to exclude eval traffic
- `schema.sql` must not drift from the SQLAlchemy model
- migration statements must be re-runnable (`init_db` runs on every startup)

Verified end-to-end against the live database: a default `log_conversation`
lands as `live`, the eval path as `eval`, and `answer_truncated` persists.

Suite green (280), ruff clean. README monitoring section updated with the
1,901/1,921 figure — it is the clearest evidence the dashboard now measures what
it claims to.

---

## Done — T10: citation anchors + full re-regeneration (2026-07-30)

**T10 was scoped as "36 chunks have junk paragraph ids". The real defect was that
48% of the corpus text was filed under the wrong citation anchor.**

`section_header` updated the section path but never *closed* the open paragraph,
so any content docling did not label as a numbered list item kept appending to the
last numbered paragraph it saw. In `ebagl_2017_16` the impact assessment,
consultation-feedback tables and annexes — **536,000 characters** — all became
"paragraph 221". Every document was affected (ebagl_2016_07 83%, ebagl_2020_05
68%), and 80/800 ground-truth questions had been generated from that text.

For a project whose stated premise is that uncited answers are worthless, answers
*confidently citing the wrong paragraph* were the most serious defect found.

### The fix

- [x] **A section heading closes the open paragraph** — the core change.
- [x] **Numbered markers recognised in plain text**, not only in `list_item`
      markers; docling mislabels them in some documents (ebagl_2020_05 collapsed
      from para 3 onward).
- [x] **No fabricated anchors.** Unnumbered content gets an empty `para_id`,
      dropped from citation metadata, so an annex cites document + section rather
      than `para. ␣model`.
- [x] **Tables attach to an unnumbered paragraph** when a heading just closed one.
- [x] **`oversized_warnings` guard**, measuring *prose* only (a large table is
      legitimately large, and a guard that fires on healthy documents gets
      ignored). Zero warnings across all 7 documents after the fix.

Result, measured per document:

| doc | prose under a wrong anchor | largest paragraph |
|---|---|---|
| ebagl_2016_07 | 63% → **0%** | 305,097 → 27,002 |
| ebagl_2020_05 | 19% → **0%** | 203,463 → 18,305 |
| ebagl_2017_16 | 53% → **0%** | 536,303 → 43,639 |

Corpus: **2,001 → 2,968 paragraphs, 1,627 → 2,241 chunks (+38%)**.
`bcbs_d424_irb` more than doubled (304 → 737 paragraphs).

### The mistake I nearly shipped

My first version scored *better* on the target metric (oversized 83% → 0%) while
silently deleting **195,000 characters of tables** — the table branch required an
open paragraph, and headings now close them. Caught only because total character
count moved when it had no business moving.

**Lesson: "the metric I was optimising improved" is not evidence a change is
good.** Check the quantities the change should *not* affect. The three numbers now
tracked together are: oversized prose → 0, total chars ≈ unchanged, table chars
≈ unchanged.

### Full chain re-run (5h07m, ~$4)

re-ingest → ground truth ×2 → retrieval evals ×2 → RAG eval. All clean: zero parse
warnings, zero unresolvable ground-truth ids, no skipped configs, 100/100 answered
and judged in every RAG config.

**Conclusions that survived** a corpus rebuilt from scratch with 38% more chunks:
- default `hybrid_rerank | structure | off` — and it now wins **both** ground
  truths (0.902 / 0.7161), so the choice no longer depends on trusting the
  de-biased set. Strictly stronger than before.
- `structure > naive` **12/12** on both sets.
- `glossary > llm` **8/8** on both sets.

**Conclusions that changed** (all now in the README, with tests updated):
- **No longer a full rank reversal.** `hybrid_rerank` tops both sets; BM25 falls
  from 2nd to last, losing 35pt between them against 19pt for the winner.
- **`off ≥ glossary` is 6/8 on the de-biased set**, not 8/8 — margins of 0.3pt,
  and expected: only 3% of those questions contain an acronym, so that arm has
  almost no signal to carry a sign. Claim split into the robust half
  (`glossary > llm`) and the measurable-only-where-acronyms-occur half.
- **gpt-4o now leads relevance** (0.89 vs 0.85) where it previously tied. The
  default stays `gpt-4o-mini` + `v2` as an explicit cost trade — 4pt of relevance
  for 15× the price, with **identical** citation support (0.78). The README now
  states plainly that the default is not the top scorer, and a test enforces that
  disclosure.
- **`v2`'s value is citations, not relevance**: +14pt cite-ok, +1pt relevance on
  gpt-4o-mini, and *nothing* on gpt-4o (which already has the discipline).

### Latency was contaminated — and is no longer published

`avg_latency_ms` made the cheap model look 7× slower than the expensive one:

| model | mean | median | p95 | max |
|---|---:|---:|---:|---:|
| gpt-4o | 2,215 | 1,982 | 3,590 | 7,723 |
| gpt-4o-mini | **17,940** | **3,291** | 12,187 | **606,291** |

A single 606-second stall moved a config's mean from ~3s to ~18s. Stalls appear
only in configs running at concurrency 4; gpt-4o ran serialised and shows none —
so the column measured harness queueing, not model speed.

- [x] `eval_rag` now records `p50_latency_ms` and `p95_latency_ms`
- [x] Latency removed from the README, with the reason stated and a test
      (`test_readme_does_not_publish_contaminated_latency`) keeping it out
- [ ] **New:** investigate the multi-minute stalls themselves — likely SDK backoff
      or queueing behind the lock-serialised cross-encoder. The percentiles hide
      the symptom; they do not explain it.
- [ ] **New:** `eval_rag` never calls `warm_query_embeddings`, so unlike
      `eval_retrieval` it still issues one single-item embedding request per
      question. Minor, but an inconsistency with T7.

Suite green (264), ruff clean. Drift guards re-verified on both tables.

---

## Done — T6 + T4 (RAG half): full 2×2 grid (2026-07-30)

`EVAL_MODELS=gpt-4o-mini,gpt-4o` judged by `gpt-5.4-mini` (cross-family against
both), n=100. **All four configs answered and judged 100/100 with zero parse
errors and zero answer errors.** The "multiple models evaluated" criterion is now
backed by an artifact.

| model | prompt | RELEVANT | cite-ok | cost/q |
|---|---|---:|---:|---:|
| gpt-4o-mini | v1 | 0.86 | 0.71 | $0.00040 |
| **gpt-4o-mini** | **v2** | **0.91** | **0.81** | **$0.00041** |
| gpt-4o | v1 | 0.90 | 0.81 | $0.00643 |
| gpt-4o | v2 | 0.88 | 0.80 | $0.00680 |

### The prompt matters more than the model

Three configs were measured twice (the first run was cut short by quota), giving a
**noise floor of ~2–3pt relevance / ~3–6pt citations**. Against that:

- `v1`→`v2` on gpt-4o-mini: **+5pt relevance, +10pt citations** — outside noise,
  same direction in both runs. Real.
- gpt-4o vs gpt-4o-mini+v2: **+1pt / 0pt at 16× the cost** — inside noise. In the
  earlier run gpt-4o was +4pt relevance and −6pt citations, which is the
  signature of noise, not an effect. gpt-4o buys nothing here.
- `v2` slightly *hurts* gpt-4o (0.90→0.88): few-shot scaffolding helps the weaker
  model and mildly constrains the stronger one. A prompt win does not
  automatically transfer across models.

Being able to say which differences are real — rather than ranking four numbers —
came free from having repeated three configs. Worth doing deliberately next time.

### Two bugs the quota exhaustion exposed

The first attempt hit `insufficient_quota` mid-run. Both were my own defects:

- [x] **Quota exhaustion was treated as transient.** It arrives as HTTP 429 like a
      TPM limit, so the per-question handler skipped ahead through 47 doomed
      questions and wrote a **biased 53-question subsample**. Now
      `QuotaExhausted` aborts the run, writes what completed, and says why.
      Detected by message — the API returns no machine-readable code for it.
- [x] **`BEST` could be won by an incomplete sample.** Now only configs with
      `answered == n` are eligible, and excluded ones are named.
- [x] A test written in T2 made **real API calls**, so it failed when credit ran
      out. Stubbed and hermetic now — it should never have depended on billing.

### Also fixed: the TPM thundering herd

Before the pacing fix, both gpt-4o configs were lost to 429s. OpenAI meters TPM
using `max_tokens` (not actual usage), so each answer reserves ~3.6k tokens →
gpt-4o's 30k TPM sustains ~8 req/min, while 3 workers attempted ~45. Every
retry's freed budget was eaten by a sibling worker, so backoff never converged.
`MODEL_MAX_CONCURRENCY = {"gpt-4o": 1}` serialises it; both configs then
completed cleanly.

- [x] README RAG section written from the CSV; the artifact-status placeholder is
      gone and 6 new tests pin the RAG table, winner, and completeness claim.
      Verified the guard bites (0.95 vs 0.91).
- [x] Suite green (250), ruff clean.

**T4 is now complete** — every evaluation artifact regenerated under the corrected
harness.

---

## Done — T5: README evaluation section rewritten (2026-07-29)

Every number in the evaluation section now comes from a committed CSV cell, and a
test suite enforces that permanently.

### Rewritten

- **New "What counts as a hit" subsection.** The relevance rule is stated *before*
  the numbers, including why document+paragraph matching fails on this corpus
  (6.7 chunks admitted per question, worst case 51). A reviewer can now judge
  whether the metric is sound, which is the first thing a sceptical reader asks.
- **Retrieval table regenerated**: 24 configs, both ground truths, with the
  rank *reversal* spelled out rather than just a winner. Corrected the two
  lexical-overlap figures — recomputed from the current ground truth as **0.81**
  and **0.45** (README said 0.83 / 0.46).
- **Three-level rewrite ablation** replaces the old on/off sentence, with the
  monotone result (8/8) and a dedicated subsection explaining why acronym
  expansion was dropped — including the conditional measurement and the honest
  caveat that this ground truth cannot represent the case where it would help.
- **structure > naive in 12/12**, plus the re-ranking-as-insurance finding
  (naive 0.825 → 0.896 on the standard set).
- **`ENABLE_REWRITE` → `REWRITE_MODE`** in the env table.

### The RAG section: no numbers it cannot support

The old table quoted 4 configs × 100 questions including two `gpt-4o` rows that no
committed file has ever contained. Removed. In its place: the methodology (which
is genuinely strong — cross-family judge, family-based `self_judged`, PARSE_ERROR
excluded from denominators) plus an explicit **artifact-status note** stating that
the committed `rag_eval.csv` is a pre-judge-swap pilot (n=10, one answer model),
is not comparable to the current harness, and is therefore not quoted. Needs one
more pass once `make eval-rag` runs.

### Drift can't recur silently: `tests/test_readme_claims.py`

15 tests parse the README and compare against the CSVs:

- every cell of the headline table, including that the "change" column is the
  actual signed difference
- the standard and de-biased winners, and both rank orders, derived from the data
- the 24-config, 8/8 and 12/12 claims re-derived from the CSVs
- the judge model and shipped `REWRITE_MODE` read from live config
- provenance guards: the RAG section must carry the artifact-status note, and the
  fabricated `gpt-4o` rows must not reappear

Verified the guard bites: re-inserting the old `0.950` fails with
`assert 0.95 == 0.9213 ± 0.001`. Chosen over a table generator because it catches
drift without owning the prose.

Suite green (239), ruff clean. Stale comments in `app/config.py` and
`.env.example` updated to the regenerated numbers.

---

## Done — T4 (retrieval half): artifacts regenerated (2026-07-29)

Both retrieval evaluations re-run at 800 questions × 24 configs under the
corrected relevance rule (T1), with eval arms that are the production paths (T2).
Clean runs: no skipped configs, no stale-ground-truth warnings.

`evaluation/results/retrieval_eval.csv` and `retrieval_eval_hard.csv` are now
trustworthy for the first time. Both carry the `rewrite_mode` column.

### Headline: the lexical bias, measured properly

Structure chunks, no rewriting:

| retrieval mode | standard hit@5 | de-biased hit@5 | collapse |
|---|---:|---:|---:|
| bm25 | **0.921** | 0.585 | **−0.336** |
| vector | 0.807 | 0.715 | −0.093 |
| hybrid | 0.892 | 0.724 | −0.169 |
| hybrid_rerank | 0.915 | **0.743** | −0.172 |

**The rank order fully reverses:**
- standard: `bm25 > hybrid_rerank > hybrid > vector`
- de-biased: `hybrid_rerank > hybrid > vector > bm25`

BM25 leads the standard set by 0.6pt and trails the de-biased set by 15.8pt.
Pure vector search is the most *robust* to paraphrase (−0.093) but never the
best. This is a far sharper version of the README's existing argument, and it now
rests on a relevance rule that doesn't inflate itself.

### Confirmed defaults — no change required

`BEST on the DE-BIASED ground truth: hybrid_rerank | structure | off`
(hit@5 **0.7425**, mrr@5 **0.5692**), which is exactly what `.env` and
`.env.example` already specify.

### Ablations, both ground truths

- `off ≥ glossary ≥ llm` in **8/8** mode×chunker triples on *both* sets.
  Rewriting is never beneficial and the harm is monotone in how much of it you
  do: ~1pt for glossary, ~15pt for the LLM step.
- `structure > naive` in **12/12** mode×rewrite comparisons on *both* sets. No
  exceptions — previously the naive chunker was flattered by recurring paragraph
  numbers.
- Reranking is insurance against bad chunking: on the standard set it lifts
  naive from 0.825 (hybrid) to 0.896, recovering most of the chunking deficit.
- Glossary's effect on the de-biased set is **+0.001** — a no-op, because only 3%
  of those questions contain an acronym. Vindicates measuring it conditionally
  in T2 rather than trusting the aggregate.

### Two independent cross-checks that the harness is honest

- `bm25|structure|off` = 0.921/0.829 and `bm25|naive|off` = 0.882/0.759 match
  the T1 predictions **exactly** (measured there by rescoring the same retrieval
  results under the new rule).
- `pre-embedded` counts confirm the acronym rates measured in T2: 1800 on the
  standard set (800 + ~200 + 800 → 25% acronym-bearing) and 1627 on the
  de-biased set (800 + 27 + 800 → the 27 I counted, 3%).

### Supersedes an earlier note in T5

T5's drift table said hybrid_rerank had overtaken BM25 on the standard set
(0.9437 vs 0.9287). That was an artifact of the **old inflated rule**, which
flattered the reranker. Under the corrected rule BM25 wins the standard set
again (0.921 vs 0.915), so the README's original "BM25 wins the standard eval,
and that win is an artifact" narrative is **restored**, not broken.

**Still open in T4:** the RAG half (`make eval-rag`), which needs the T6 decision
on the answer-model arm first.

---

## Done — T7 + T8: evaluation performance (2026-07-29)

### T7 — query-embedding cache + batch warm-up: **1440 → 1 HTTP request**

`embed_query` issued one single-item request per vector/hybrid search, so the
same query string was re-embedded once per configuration. Now cached (bounded
LRU, keyed by model+text, returns a copy so callers can't poison it), and
`warm_query_embeddings` pre-embeds every query in batches of 128 before the
configs run.

Measured on 80 query strings × 18 embedding-using configs:

| | HTTP requests | texts embedded | time |
|---|---:|---:|---:|
| before | 1440 | 1440 | 238.3 s |
| after | **1** | **41** | **1.2 s** |

(41 not 80 because on the de-biased set `off` and `glossary` produce identical
queries for the 97% of questions containing no acronym.) The eval ran embedding
calls 8-way parallel, so real wall-clock saving is ~8× less than the sequential
200× — but the request-count reduction is exact.

### T8 — reranker threading: **no local effect, 2.38× in Docker**

The hypothesis was right but incomplete. `torch.set_num_threads(1)` was added
before `_rerank_lock` existed; once reranking is serialised the cap can only
waste cores. But locally it changed nothing:

| torch threads | ms/query (interleaved runs) |
|---|---|
| 1 | 505, 502 |
| 12 | 521, 491 |

**Why:** `CrossEncoder` auto-selects a device, and on Apple Silicon that is MPS —
the reranker was already on the GPU, so CPU thread count was irrelevant. Forcing
`device="cpu"` shows the reranker is 99% of `hybrid_rerank` (470 of 476 ms) and
that threads matter a lot there:

| device / threads | ms per rerank (20 pairs) |
|---|---:|
| cpu, 1 thread | 2602 |
| cpu, 4 threads | 1372 |
| cpu, 12 threads | **1094** |
| mps (local default) | **469** |

So the change is worth keeping — it is a **2.38× speedup in the CPU-only Docker
image and on Linux/CI**, and a no-op locally. `limit_torch_threads` renamed to
`configure_torch_threads`, with the real rationale recorded.

### Robustness fix surfaced along the way

The first `--limit 40` run died with
`ResponseHandlingException: [Errno 9] Bad file descriptor` from a pooled Qdrant
connection, and **took all 23 completed configs with it**. Intermittent (later
runs were clean), but a full 800×24 run is long enough to make it likely.

- [x] `Retriever._qdrant_search` retries transport-level failures (3 attempts,
      backoff). Only `ResponseHandlingException` — real API errors
      (`UnexpectedResponse`: missing collection, bad filter) still fail fast.
      Also stops a transient blip becoming a spurious 502 for a user.
- [x] **`eval_retrieval` gained the per-config resilience `eval_rag` already
      had** — it had none, so one blip discarded the whole run. Skipped configs
      are logged, completed ones still written, and a count is printed. This is
      the same lesson as commit 87fc810, never applied to the retrieval eval.

- [x] Tests: cache hit/copy/eviction/model-keying, batch dedup, thread config,
      Qdrant retry-then-succeed and give-up. Suite green (223), ruff clean
- [x] Verified end-to-end: two clean `--limit 40` runs, all 24 configs

---

## Done — T3: judge parse failures are missing data, not verdicts (2026-07-29)

`parse_judge` defaulted any malformed, truncated or unrecognised-label response
to `NON_RELEVANT` + `citations_supported=False`. That is indistinguishable from
a genuine negative verdict, so it pushed every config's score down — and hardest
for whichever config made the judge most verbose, a bias that looks exactly like
a quality difference.

**Now:** a distinct `PARSE_ERROR` outcome that is *not* a relevance label, with
`JudgeResult.parsed` to test against. Quality rates are computed over `judged`
(= n − parse_errors), so an unreadable verdict shrinks the sample instead of
counting as a bad answer. Cost and latency stay over all n — they were really
spent.

- [x] `PARSE_ERROR` + `JudgeResult.parsed`; an unknown label is a parse error
      too (it used to silently become NON_RELEVANT)
- [x] `judge_answer` retries once, then reports `PARSE_ERROR`, billing **every**
      attempt so cost accounting stays honest
- [x] Failure cause printed per attempt (`truncated at the token cap` vs
      `unparseable response`, with `finish_reason` and output tokens), so a
      systematic problem is diagnosable instead of just a lower score
- [x] `finish_reason` + `LLMResult.truncated` on both `complete` and
      `stream_complete` — this was the missing signal
- [x] Judge budget 200 → 600 tokens, and the prompt caps `reason` to one short
      sentence placed last. 200 was tight: the judge is asked for free text, and
      a reasoning-model judge spends part of the budget thinking before emitting
      anything
- [x] New CSV columns `judged` and `parse_errors`; per-config WARNING when any
      verdict is unreadable, so a shrunken denominator is never silent
- [x] Parse errors plotted as a grey top segment — a judge failure is visible in
      the artifact rather than just making the bar shorter
- [x] Unreadable verdicts log `judge_relevance = NULL` (not "PARSE_ERROR"), so
      the Grafana relevance panel shows real judgements only
- [x] Tests: exclusion arithmetic, all-errors edge case, NULL logging, retry
      success, retry exhaustion, truncation detection. Suite green (214)
- [x] Live smoke test: real judge returns RELEVANT/parsed in 1100ms at $0.000426;
      `eval_rag --n 2` writes the new schema with 0 parse errors

**Partly addresses T14** (`finish_reason` ignored): the provider now surfaces it.
What remains is propagating a `truncated` flag onto `Answer` so a user-facing
answer cut off at `MAX_ANSWER_TOKENS` is visibly marked — that needs a
`conversations` column, so it is left with T11's migration.

---

## Done — T2: eval arms now ARE the production paths (2026-07-29)

The eval's "rewrite off" arm used the raw question, but production with
`ENABLE_REWRITE=false` still applied glossary expansion — so the shipped default
had never been measured. Fixing the arm exposed a bigger problem: the old
boolean **conflated "no LLM rewrite" with "no rewriting at all"**, which meant
acronym expansion ran unconditionally in production and could not be ablated.

**Replaced `ENABLE_REWRITE` with `REWRITE_MODE` = `off` | `glossary` | `llm`.**
Every eval arm is now produced by calling the production `rewrite_query` under a
settings copy, so the arms cannot drift from the shipped code path again. The
retrieval eval goes 16 → 24 configs.

### The finding: acronym expansion was silently degrading retrieval

Aggregate deltas looked like noise, so I measured **conditional on the queries
where expansion actually changes the query** (n=200, standard set):

| retrieval | Δ hit@5 | Δ mrr@5 |
|---|---:|---:|
| bm25 | **−0.035** | **−0.071** |
| vector | **−0.050** | **−0.040** |

`expand_acronyms` appends "LGD (loss given default)", which dilutes BM25 term
weights and shifts the embedding *away* from the corpus's own vocabulary — the
regulations use the acronyms too. **Default flipped to `off`.**

**Why the aggregate hid it:** only 25% of standard-set questions contain a
glossary acronym, and just **3% of the de-biased set** — paraphrasing away
passage vocabulary also paraphrased away the acronyms. So the de-biased ground
truth, the set used to pick the default, is structurally incapable of evaluating
acronym expansion. Its ~0.000 delta was uninformative, not evidence of no effect.
Any future feature that only fires on a query subset needs the same conditional
treatment.

**Caveat, stated honestly:** ground-truth questions are generated *from* the
passages, so they inherit the corpus's acronym usage. A real user who types
"MoC" against a document that spells out "margin of conservatism" is the case
where expansion should help, and this ground truth cannot represent it. The
measured harm is real; the measured benefit may be understated. `off` is the
evidence-backed default, revisitable with real query logs (the monitoring DB now
collects them).

- [x] `RewriteMode` literal + `REWRITE_MODE` in `app/config.py`
- [x] `rewrite_query` honours all three levels (`app/rewrite.py`)
- [x] `precompute_queries` builds every arm via `rewrite_query` — eval == production
- [x] Legacy `ENABLE_REWRITE` still honoured (`false`→`glossary`, `true`→`llm`,
      matching what it actually did) so no deployment changes behaviour silently;
      declared as a real field so it is picked up from `.env`, not just the env
- [x] CSV column `rewrite` → `rewrite_mode`; labels and BEST line updated
- [x] Tests: three modes, off≠glossary, legacy migration, and an eval-arm ==
      production-path assertion. Suite green (206), ruff clean
- [x] README: `ENABLE_REWRITE` → `REWRITE_MODE` (variable rename only; the
      evaluation tables stay untouched until T4/T5)

---

## Done — T1: relevance rule tightened (2026-07-29)

**Implemented differently from the plan below, and better.** The plan proposed
`chunk_id` match OR (same doc + section path + paragraph overlap). Built and
measured, that only got the mean from 6.71 to 4.28 accepted chunks per row —
because it cannot separate parse failures where one "paragraph" spans many
chunks inside a single section (`ebagl_2017_16` para. 221 is **51 chunks**, all
under "9 Review of estimates").

**Shipped rule — one criterion for both chunkers:** exact `chunk_id`, OR same
document AND the chunk reproduces ≥30% of the ground-truth passage's word
5-grams (`relevant_by_text`). It asks the only question that matters — *does
this chunk contain the passage the question was written from?* — and is immune
to both recurring paragraph numbers and the mega-paragraph parse failures.
Using an identical rule for structure and naive is also what makes the
comparison fair; neither is judged more leniently.

Ambiguity collapse (chunks accepted per ground-truth row):

| rule | structure | naive |
|---|---:|---:|
| old (doc + paragraph) | 6.71 (max 51) | — |
| planned (+ section path) | 4.28 (max 51) | — |
| **shipped (text coverage)** | **1.10** (max 3) | **1.58** (max 4) |

Measured effect at full scale (800 questions, BM25, same retrieval results
scored both ways — so this isolates the rule change):

| ground truth | chunker | old hit@5 | new hit@5 | Δ |
|---|---|---:|---:|---:|
| standard | structure | 0.929 | 0.921 | −0.008 |
| standard | naive | 0.902 | 0.882 | −0.020 |
| de-biased | structure | 0.613 | 0.585 | −0.028 |
| de-biased | naive | 0.589 | 0.530 | **−0.059** |

**The headline numbers were only mildly inflated — but the comparison was
materially biased.** The old rule favoured the naive chunker roughly 2–3× more
than structure, because a 500-token window sweeps up more recurring paragraph
numbers. On the de-biased set the structure-over-naive advantage more than
doubles: 2.4pts → 5.5pts. Structure-aware chunking is better-justified than the
old numbers showed.

- [x] `relevant_by_text` / `shingles` / `text_coverage` in `evaluation/metrics.py`
- [x] `GroundTruthIndex` resolves rows to their source chunk + caches shingles
- [x] `relevant_by_paragraph` kept ONLY as the stale-ground-truth fallback, now
      documented as unsafe alone, with a loud `report_missing()` warning
- [x] Threshold 0.30 calibrated: the highest value at which no ground-truth row
      becomes unreachable for either chunker (an unreachable row is a guaranteed
      miss that silently penalises that chunker)
- [x] `tests/test_eval_methodology.py` rewritten — same-paragraph-number-
      different-passage is now a **miss**; a wider chunk containing the passage
      is still a **hit**
- [x] Text-coverage unit tests in `tests/test_eval.py`; suite green (198), ruff clean
- [x] **T15 resolved by deletion** — the unused `chunker` parameter is gone;
      one uniform rule means `evaluate_config` no longer needs it
- [x] Verified end-to-end: `eval_retrieval --limit 60` runs all 16 configs

**Not fixed by T1 (upstream data quality, still open):** `ebagl_2017_16` para.
221 spanning 51 chunks and `ebagl_2016_07` para. 114 spanning 38 is a *parse*
failure — a paragraph marker was missed and one "paragraph" swallowed a large
span. The relevance rule no longer cares, but citations for those spans are
still wrong for users. Folded into **T10**.

---

## Done — judge swap + provider cross-family fixes (2026-07-29)

Motivated by hitting the gpt-4o TPM wall. Measured demand: a full `eval_rag`
n=100 × 4 configs is **2.09M tokens, of which 1.12M is judge traffic on a single
model** — so the judge, not the answer model, was the bottleneck.

- [x] **Judge swapped gpt-4o → gpt-5.4-mini** (`JUDGE_MODEL` in `app/config.py`,
      `.env`, `.env.example`; `--judge-model` now overrides config rather than
      hardcoding a default). 30k → 500k TPM, $7.92 → $2.56 per 1k judgments,
      full-run cost $3.38 → $1.23, still cross-family vs gpt-4o-mini answers.
- [x] **`max_completion_tokens`** replaces the legacy `max_tokens`
      (`app/providers.py:_request_kwargs`). Verified: every GPT-5 model 400s on
      `max_tokens`; GPT-4o accepts the new name, so one code path serves both.
- [x] **Per-model temperature handling** (`FIXED_TEMPERATURE_MODELS`). Verified:
      gpt-5-mini rejects `temperature=0.0`; gpt-4o-* and gpt-5.4-* accept it.
- [x] **PRICING extended to the GPT-5 family**, plus `require_pricing()` called
      at the top of `eval_rag.main()` so an unpriced model fails fast instead of
      silently recording $0 costs.
- [x] **Reasoning tokens recorded** on `LLMResult` (already inside `tokens_out`
      and priced at the output rate — captured for visibility, never
      double-counted).
- [x] **Self-preference detection is now family-based**, not name equality — a
      gpt-5.4 judge scoring gpt-5.4-mini answers is the same bias.
- [x] `tests/test_providers.py` added; suite green (187 tests, up from 161),
      ruff clean.
- [x] Live smoke test: judge returns RELEVANT / cite_ok in 1390ms at $0.000481.

**Models assessed and rejected for this use case:**
- `gpt-5-mini` — rejects `temperature=0.0` (loses eval reproducibility) and
  spends 192 of 242 output tokens on reasoning by default.
- `gpt-5-nano` — 384 of 435 output tokens reasoning, 6.5s latency; slower and
  more token-hungry than gpt-5-mini despite the name.
- `gpt-5.4-mini as answer model` — good (temperature=0 OK, 0 reasoning tokens,
  ~850ms vs gpt-4o-mini's 1651ms) but 6× the cost for a task where gpt-4o-mini
  already scores 0.9. Worth revisiting only as the *second* model for T6.

---

## Phase 1 — evaluation methodology (blocks everything downstream)

> **Phase 1 and Phase 2 are complete** (T1–T8 + T15). Every evaluation artifact
> has been regenerated under the corrected harness and the README matches it,
> enforced by `tests/test_readme_claims.py`. What remains is Phase 4/5:
> data quality (T9, T10), monitoring hygiene (T11), the Batch API option (T18),
> and small cleanups.

### [x] T1. Tighten the retrieval relevance rule (HIGH) — DONE, see above

**Where:** `evaluation/metrics.py:68` (`relevant_by_paragraph`),
`evaluation/eval_retrieval.py:96` (`_is_relevant`)

**Problem:** a hit is *same `doc_id` + any shared paragraph number*. But
`ingestion/models.py:51` documents that paragraph numbering restarts in every
section, so one paragraph number maps to many unrelated chunks.

Measured on the current corpus (1627 structure chunks, 800 GT rows):

| metric | value |
|---|---|
| distinct `(doc_id, para_id)` keys | 1109 |
| keys owned by >1 chunk | 405 |
| GT rows with >1 chunk accepted as relevant | **544 / 800 (68%)** |
| mean chunks accepted per GT row | **6.71** |
| worst case (`ebagl_2017_16` para `"221"`) | **51 chunks** |

hit-rate@5 and MRR@5 are therefore inflated, unevenly across configs — and the
default-config decision rests on them.

**Fix:** relevance = exact `chunk_id` match **OR** (same `doc_id` AND paragraph
overlap AND same `section_path`). For naive chunks, substring-containment of the
ground-truth chunk text.

**Acceptance:**
- [ ] Re-run the ambiguity script: mean accepted chunks per GT row ≈ 1.0–1.5
- [ ] `tests/test_eval_methodology.py::test_relevance_is_symmetric_paragraph_overlap`
      updated — a same-number-different-section chunk must now be a **miss**
- [ ] New test: hard-split sibling of the GT chunk is still a **hit**

**Follow-up (optional, larger):** section-qualify paragraph ids
(`"5.3.2/82"`) in `ingestion/parse.py` so the ambiguity disappears at the source
and citations become unique. Requires re-ingest + ground-truth regeneration.

### [x] T2. Make the "rewrite off" eval arm match production (HIGH) — DONE, see above

**Where:** `evaluation/eval_retrieval.py:88` (`precompute_queries`)

**Problem:** the `False` arm uses the raw question, but production
(`app/rewrite.py:111`, `ENABLE_REWRITE=false`) returns
`expand_acronyms(question)` — it always appends `[PD (probability of
default); …]`, which changes both BM25 and the embedding. The shipped default
config has never actually been measured.

**Fix:** build the arm with `rewrite_query(q, settings_with_rewrite_off).rewritten`.

**Decide while here:** should `ENABLE_REWRITE=false` also disable glossary
expansion? SPEC §7 describes the glossary as part of rewriting. Either gate it
behind the flag or document that the flag only controls the LLM call.

**Acceptance:**
- [ ] Both eval arms route through `rewrite_query`, differing only by the setting
- [ ] Test asserting the "off" arm's query equals what `app.rag._prepare` builds

### [x] T3. Stop scoring judge parse failures as NON_RELEVANT (HIGH) — DONE, see above

**Where:** `evaluation/judge.py:44` (`parse_judge`), `app/providers.py:85`

**Problem:** malformed/truncated judge JSON silently becomes
`NON_RELEVANT` + `citations_supported=False`. The judge runs at
`max_tokens=200` while being asked for a free-text `reason`, so truncation is
plausible and indistinguishable from a genuine negative verdict. Biases every
config down, verbose-judgment configs more.

**Fix:**
- [ ] Distinct `PARSE_ERROR` outcome; count it and exclude from the denominator
      (or retry once)
- [ ] Surface `finish_reason` from `providers.complete` so truncation is visible
- [ ] Raise the judge token budget or drop the `reason` field
- [ ] Report a `parse_error` column in `rag_eval.csv`

### [x] T4. Regenerate evaluation artifacts — DONE (retrieval + RAG)

Only after T1–T3 land.

- [x] `make eval-retrieval` (standard) + `make eval-retrieval-hard` — 24 configs
      each, 800 questions, clean. See the T4 section above.
- [x] `make eval-rag` — 4 configs, n=100, all complete
- [x] BEST confirmed = `hybrid_rerank | structure | off`, which `.env` and
      `.env.example` already specify — no change needed

---

## Phase 2 — reproducibility / rubric exposure

### [x] T5. Re-sync README (and code comments) with the committed CSVs (HIGH) — DONE, see above

**Where:** `README.md:358-397`, `app/config.py:63-66`, `.env.example:22`

The rubric asks for tables copied from `results/`. Current drift:

| claim | README | committed CSV |
|---|---|---|
| standard, bm25/structure | 0.950 ("wins") | 0.9287 |
| standard, hybrid_rerank | 0.936 | 0.9437 (old rule) → **0.915** regenerated; BM25 wins the standard set again — see T4 |
| de-biased bm25 / vector / hybrid / hybrid_rerank | 0.528 / 0.605 / 0.615 / 0.636 | 0.6125 / 0.7475 / 0.7475 / 0.7688 |
| RAG eval | 4 configs × 100 questions incl. gpt-4o | 2 rows, n=10, gpt-4o-mini only |

Note the **narrative** breaks, not just the digits: README:362 argues
"`bm25/structure/raw` wins (0.950) … but that win is a measurement artifact".
On current numbers BM25 doesn't win the standard set — `hybrid_rerank` does.
The conclusion (hybrid_rerank leads the de-biased set) survives; the argument
needs rewriting.

**Acceptance:**
- [x] Every README number traceable to a cell in `evaluation/results/*.csv`
- [x] Lexical-overlap figures (0.83 / 0.46) re-checked against the regenerated
      `*.meta.json` / generator output
- [x] Stale comments in `app/config.py` and `.env.example` updated
- [x] README:386 still says "A judge (gpt-4o)" — the judge is now
      `gpt-5.4-mini` (see Done). Update the prose and explain the choice.
- [x] `self_judged` is now family-based; describe what the column means
- [x] Drift prevention: `tests/test_readme_claims.py` compares README numbers to
      the CSVs (chosen over a table generator — catches drift without owning the
      prose). Verified it fails on the old stale value.

### [x] T6. Back the "multiple models evaluated" criterion with an artifact (HIGH) — DONE, see above

**Where:** `.env:10` (`EVAL_MODELS=gpt-4o-mini`), `evaluation/results/rag_eval.csv`

The gpt-4o answer arm was dropped for the 30k-TPM limit, so the committed CSV
varies only prompt v1/v2 while the README asserts gpt-4o rows.

Pick one:
- [ ] Re-run with `EVAL_MODELS=gpt-4o-mini,gpt-4o` at reduced `--n` / `--workers`
      so it fits the rate limit, **or**
- [ ] State plainly in the README that the model arm was dropped, why, and what
      evidence remains

Also: the committed run is `--n 10`; SPEC §11.3 specifies 100. Re-run at 100 or
document the deviation.

---

## Phase 3 — performance

### [x] T7. Cache query embeddings (MEDIUM) — DONE, see above

**Where:** `app/retrieval.py:174` → `app/providers.py:161` (`embed_query`)

One HTTP round-trip per search, batch-of-one. In `eval_retrieval`, 12 of 16
configs use embeddings × 800 questions = **9,600 calls for 1,600 distinct query
strings**.

- [ ] `lru_cache` keyed on `(embedding_model, text)`, **or** precompute+batch
      (128/request) alongside `precompute_queries`
- [ ] Verify ~6× fewer embedding requests on a `--limit 50` run

### [x] T8. Let the cross-encoder use more than one core (MEDIUM) — DONE, see above

**Where:** `evaluation/corpus.py:27` (`limit_torch_threads`),
`app/retrieval.py:206` (`_rerank_lock`)

`torch.set_num_threads(1)` guards against oversubscription, but `_rerank_lock`
already serializes `predict()` — so exactly one rerank runs at a time, on one
core, while the rest idle. The lock makes multi-threaded torch safe here. This
is the eval's long pole (~64k query-passage pairs across the 4 rerank configs).

- [ ] Raise the torch thread cap (keep `TOKENIZERS_PARALLELISM=false`)
- [ ] Time a `--limit 50` hybrid_rerank run before/after; record the speedup

### [ ] T18. Move the evaluations to the Batch API (MEDIUM — structural)

The judge swap (see Done) bought 16× TPM headroom, but the evaluations are
inherently offline and latency-insensitive — the textbook Batch API workload.
Tier 1 allows a 5M-token batch queue for gpt-5.4-mini, which sidesteps TPM
entirely and halves token cost.

Do this if the eval grows (n=100 → larger, or more configs); the judge swap is
sufficient for the current size.

- [ ] Batch-submit `eval_rag` answer+judge calls and `generate_ground_truth`
- [ ] Async result collection + resumability (batches complete out of band)
- [ ] Keep the synchronous path for `--limit`/smoke runs

---

## Phase 4 — correctness / data quality

### [ ] T9. Naive chunker welds paragraphs together (MEDIUM)

**Where:** `ingestion/chunk.py:177` (`chunk_naive`)

Per-paragraph token ids are concatenated with no separator. Verified:

```
'Institutions shall apply a margin of conservatism.The margin shall reflect estimation error.'
```

Corrupted token boundaries at every junction, in the baseline that is supposed
to be a *fair* comparison.

- [ ] Insert `encode("\n\n")` between paragraphs
- [ ] Test asserting no welded `.`-to-capital junctions
- [ ] Rebuild the naive index (`make eval-retrieval` with `--rebuild-naive`)

### [x] T10. Junk paragraph ids become citation anchors (MEDIUM) — DONE, see above

> Scope turned out to be far larger than described: 48% of corpus text was under
> the wrong anchor, not 36 chunks. See the T10 section at the top.

**Where:** `ingestion/parse.py:172` — `para_id=marker.rstrip(".") or text[:8]`

Produced ids like `"<U+F0A7> model "` (a Wingdings bullet in the PDF, U+F0A7 —
private use area) and `"© Bank  "` — 36/1627 chunks. These render to users as
`[ECB Guide…, para. <U+F0A7> model ]`, and that one id alone owns 26 chunks,
feeding T1's ambiguity.

- [ ] Reject non-citation-shaped markers; fold that text into the current paragraph
- [ ] **Also from T1:** missed paragraph markers let one "paragraph" swallow a
      huge span — `ebagl_2017_16` para. 221 covers 51 chunks, `ebagl_2016_07`
      para. 114 covers 38. Citations for those spans point users at the wrong
      text. Assert no paragraph spans more than a few chunks after re-parsing.
- [ ] Re-parse; assert 0 non-numeric / non-`Article`-shaped `para_id`s
- [ ] Requires re-ingest → regenerate ground truth (chunk ids change)

### [x] T11. Separate eval traffic from live traffic in monitoring (MEDIUM) — DONE, see above

**Where:** `evaluation/eval_rag.py:111` (`_safe_log`), `monitoring/db.py:35`,
`monitoring/schema.sql`

Eval answers land in `conversations` unmarked, so Grafana's "questions per day",
"cost over time" and "latency p50/p95" mix synthetic runs with real usage.

- [ ] Add a `source` column (`'live' | 'eval'`, default `'live'`) + the
      `ADD COLUMN IF NOT EXISTS` migration in `_ADDED_COLUMNS`
- [ ] Filter the usage/cost/latency panels to `source = 'live'`; keep the judge
      relevance panel on eval rows

---

## Phase 5 — low severity

- [ ] **T12** `app/rag.py:152` — `check_citation_grounding` unions paragraphs
      across *all* title-matching docs. Verified: `"EBA Guidelines on LGD
      estimation"` matches 2 corpus documents, so a citation can be "grounded"
      by a paragraph from a different document. (No false matches between the 7
      full titles themselves.) Scope the union to the single best-matching title.
- [~] **T13** `para. 82-85` is no longer *always* flagged — the sub-point
      normalisation added for the `n/a` fix also yields `82` as an accepted
      form, so a range is grounded when its **first** paragraph was retrieved
      (verified). Still imperfect: `82-85` backed only by retrieved paragraph 84
      is flagged. Expanding the full range would fix it, but that loosens the
      check further, so measure how often ranges actually occur first.
- [x] **T14** DONE — `finish_reason` surfaced in T3; `Answer.truncated`,
      `conversations.answer_truncated`, the UI warning and a dashboard panel all
      landed with T11's migration.
- [x] **T15** `evaluate_config`'s unused `chunker` parameter — removed as part
      of T1 (one uniform relevance rule made it genuinely unnecessary).
- [ ] **T16** `evaluation/eval_rag.py:40` — `load_sample` is hardcoded to the
      standard (lexically biased) ground truth. Add `--ground-truth`, matching
      `eval_retrieval`.
- [x] **T17** `docker-compose.yml` header — said api/ui "are intentionally not
      defined yet" when both are defined. Rewritten to list the five services and
      the `up` / `up-all` / `prod-up` entry points; every claim verified against
      `docker compose config --services` and the Makefile.

---

## Verification gate (before considering this done)

- [ ] `make test` green (161+ tests)
- [ ] `make lint` clean
- [ ] `evaluation/results/*.csv` regenerated after Phase 1 + T9/T10
- [ ] Every README evaluation number traceable to a committed CSV cell
- [ ] `.env.example` defaults equal the winning config in the regenerated CSVs
