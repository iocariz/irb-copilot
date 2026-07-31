# IRB Copilot

A retrieval-augmented (RAG) Q&A assistant over public **EU prudential regulation
for credit-risk modeling** — the IRB (internal ratings-based) approach, the
definition of default, and loan origination. It answers questions from analysts
building or validating credit-risk models and **cites every claim by document +
paragraph**, because in this domain an uncited answer is worthless.

> **Live example.** *"How many days past due trigger a default?"* →
> *"A default is triggered … after 90 consecutive days of material past due
> amounts [ECB Guide to Internal Models (February 2024, consolidated), para. 66]."*

Built as the final project for the DataTalks.Club **LLM Zoomcamp**.

---

## Table of contents

- [Problem](#problem) · [Features](#features) · [Tech stack](#tech-stack)
- [Architecture](#architecture) · [How it works](#how-it-works) · [Project structure](#project-structure)
- [Dataset](#dataset) · **[Run it yourself](#run-it-yourself-reviewer-walkthrough)** · [Running the app](#running-the-app)
- [Configuration](#configuration) · [Evaluation](#evaluation) · [Monitoring](#monitoring)
- [Deployment](#deployment) · [Development](#development) · [Design decisions](#design-decisions--trade-offs)
- [Rubric mapping](#rubric-mapping)

---

## Problem

Credit-risk analysts building IRB models must comply with dense, overlapping EU
regulation spread across thousands of numbered paragraphs — EBA guidelines, the
ECB Guide to Internal Models, and the Basel framework. Finding the exact
provision that governs a modeling choice, and citing it precisely, is slow and
error-prone; the wording of "margin of conservatism", "downturn LGD" or
"days past due" is scattered across several documents that cross-reference each
other.

IRB Copilot retrieves the relevant paragraphs from a vetted corpus and produces a
grounded, **citation-first** answer, so every statement can be traced back to its
source paragraph — and the app tells you when it *can't* answer from the corpus
rather than guessing.

## Features

- **Grounded, cited answers** — every factual sentence ends with `[doc_title, para. X]`.
- **Answer-time citation self-check** — flags citations that don't map to a
  retrieved source as possible hallucinations.
- **Four retrieval modes** — `bm25`, `vector`, `hybrid` (reciprocal rank fusion),
  `hybrid_rerank` (cross-encoder), all swappable by env var; the default is the
  one a rigorous (de-biased) evaluation found best.
- **Streaming UI** — answers stream token-by-token, with expandable source
  snippets, 👍/👎 feedback, and follow-up questions (short conversation memory).
- **Automated ingestion** — `download → parse → chunk → index`, orchestrated with
  Prefect; docling extracts real document structure (headings, numbered
  paragraphs as citation anchors, tables).
- **Full evaluation harness** — retrieval (hit-rate@5, MRR@5) and RAG
  (LLM-as-a-judge relevance + citation support), with a de-biasing analysis.
- **Monitoring** — every question + feedback logged to Postgres, visualised in an
  8-panel Grafana dashboard (incl. a hallucination-rate trend), with evaluation
  traffic separated from real usage.
- **Containerised & deployable** — one `docker compose` for everything, plus a
  hardened VM deployment behind a Caddy TLS proxy.

## Tech stack

| Layer | Choice |
|-------|--------|
| Language / packaging | Python 3.12, [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`) |
| Vector store | Qdrant (cosine) |
| Lexical index | `bm25s` (persisted to disk) |
| LLM + embeddings | OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) via a thin provider module |
| Reranker | `BAAI/bge-reranker-base` cross-encoder (sentence-transformers, CPU) |
| PDF parsing | docling (native structure) with a pymupdf fallback |
| API / UI | FastAPI + Streamlit |
| Monitoring | Postgres (SQLAlchemy) + Grafana |
| Orchestration | Prefect (ingestion flow) |
| Tests / lint | pytest, ruff |

Every model call goes through [`app/providers.py`](app/providers.py), so the
provider or model is a one-line env change.

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion — Prefect flow"]
        DL[download<br/>sha256-verified] --> PA[parse<br/>docling → paragraphs]
        PA --> CH[chunk<br/>structure-aware]
        CH --> IX[index]
    end
    IX --> QD[(Qdrant<br/>vectors)]
    IX --> BM[(BM25<br/>lexical index)]

    subgraph Serving
        UI[Streamlit UI] -->|/ask, /ask/stream| API[FastAPI]
        API --> RAG[rag.answer<br/>rewrite? → retrieve → prompt → LLM → self-check]
        RAG --> QD
        RAG --> BM
        RAG --> LLM[(OpenAI)]
    end

    API -->|log| PG[(Postgres<br/>conversations, feedback)]
    UI -->|/feedback| API
    PG --> GF[Grafana dashboard]

    subgraph Evaluation
        GT[generate_ground_truth] --> ER[eval_retrieval]
        GT --> RE[eval_rag<br/>LLM-as-judge]
    end
```

## How it works

### 1. Ingestion (`python -m ingestion.flow`)

A four-stage pipeline; each stage reads the previous stage's on-disk artifacts,
so stages are independently runnable, resumable (`--from-stage`, `--to-stage`)
and idempotent. Prefect wraps each stage as a task (retries, logging); the same
stages run standalone via `python -m ingestion`.

1. **download** ([`ingestion/download.py`](ingestion/download.py)) — fetches each
   PDF listed in [`data/sources.yaml`](data/sources.yaml), verifies its SHA-256
   (pinning it on first download), and is idempotent. A 404 fails loudly with the
   document id — nothing is silently skipped. PDFs are never committed.
2. **parse** ([`ingestion/parse.py`](ingestion/parse.py)) — docling recovers real
   structure: section headings, **numbered list-item markers** (the citation
   anchors, e.g. `26.`), tables (kept as markdown), and footnotes (dropped as
   noise). Sub-points (`a.`, `ii.`) are folded into their parent paragraph. If
   docling is unavailable or fails on a document, a pymupdf text extraction +
   regex fallback kicks in.

   Two rules exist specifically to keep citation anchors honest. A **section
   heading closes the open paragraph** — without that, any content docling does
   not label as a numbered item keeps appending to the last numbered paragraph it
   saw, which had put annexes and consultation-feedback tables under a paragraph
   number that does not contain them. And content with no paragraph number
   (annexes, front matter) gets an **empty anchor** rather than one invented from
   its own text, so it cites document + section instead of a paragraph that does
   not exist. A `MAX_PARAGRAPH_CHARS` guard warns at ingest time if any paragraph
   accumulates implausible amounts of prose, so a regression surfaces there rather
   than in a user's citation.
3. **chunk** ([`ingestion/chunk.py`](ingestion/chunk.py)) — **structure-aware**:
   one chunk per numbered paragraph; consecutive paragraphs in the same
   subsection are merged while under ~250 tokens; any chunk over ~1000 tokens is
   split at sentence boundaries. Each chunk keeps `{doc_id, doc_title,
   section_path, para_ids, pages, text}`. Chunk ids are deterministic
   (`uuid5(doc_id + para_ids + ordinal)`) so re-runs upsert instead of
   duplicating — the ordinal is needed because regulatory docs restart paragraph
   numbering in every section. A **naive** fixed-window chunker (500 tokens, 50
   overlap) exists as the evaluation baseline.
4. **index** ([`ingestion/index.py`](ingestion/index.py)) — embeds chunk text and
   upserts into the Qdrant collection `irb_chunks`, and builds a persistent BM25
   index (+ a chunk manifest) on disk. A guardrail warns if the collection holds
   more points than were just indexed (orphans from an earlier run) and suggests
   `--recreate`.

### 2. Retrieval ([`app/retrieval.py`](app/retrieval.py))

Four modes, selected by `RETRIEVAL_MODE`, all returning the top-k chunks with
scores + metadata and supporting an optional `doc_ids` filter:

- **`bm25`** — lexical (exact-term) search over the persisted BM25 index.
- **`vector`** — Qdrant cosine similarity over OpenAI embeddings.
- **`hybrid`** — **reciprocal rank fusion** (RRF, k=60) of the bm25 and vector
  rankings: `score = Σ 1/(k + rank)`.
- **`hybrid_rerank`** — take the hybrid top-20, then re-order with a
  cross-encoder (`BAAI/bge-reranker-base`) that scores each `(query, chunk)` pair
  directly, and keep the top-k. This is the **default** (see [Evaluation](#evaluation)).

### 3. Query rewriting ([`app/rewrite.py`](app/rewrite.py))

Three levels, selected by `REWRITE_MODE`: `off` (the question verbatim),
`glossary` (expand domain acronyms — PD, LGD, EAD, MoC, DoD, CCF, CRR, RDS… —
from a hard-coded glossary, deterministic and free), or `llm` (glossary plus an
LLM reformulation). Each level is ablated separately by `eval_retrieval`.

The evaluation found rewriting *hurts* retrieval on this corpus at both levels —
expansion and paraphrase both move the query away from the exact regulatory
terms the documents use — so the default is `off`, but all three are implemented
and measured.

For **follow-up questions**, a separate step (`condense_query`, `CONDENSE_HISTORY`,
on by default) rewrites the follow-up into a *standalone* query using the chat
history — so "and what is the materiality threshold for that?" becomes a query
that resolves *that* to *past due amounts* and retrieves the right paragraphs,
instead of searching on a referent-free question. It only fires when history is
present (first-turn retrieval is unchanged).

### 4. RAG flow ([`app/rag.py`](app/rag.py))

`answer(question, doc_ids=None, history=None)` runs:

1. **resolve the retrieval query** — condense a follow-up to a standalone query
using history, or apply the optional first-turn rewrite → 2. **retrieve** top-k →
3. **build prompt** (context block with citation headers + conversation history) →
4. **LLM** → 5. **parse citations** → 6. **self-check** grounding.

The prompt ([`app/prompts.py`](app/prompts.py)) comes in two variants: `v1`
(plain) and `v2` (few-shot example + stricter citation format). Both instruct the
model to answer **only** from the provided context, to say so when the context is
insufficient, and to cite inline as `[doc_title, para. X]`.

The **citation self-check** (`check_citation_grounding`) verifies every parsed
citation maps to a retrieved source — a title-matching document where **every**
cited paragraph was actually retrieved (so a partly hallucinated `para. 82, 99`
isn't laundered by the one real paragraph). Citations that don't are surfaced as
`ungrounded_citations` and shown in the UI as a "possible hallucination" warning.

The returned `Answer` carries: `text`, parsed `citations`, `chunks_used` (with
scores + full text for the UI), `citations_grounded` / `ungrounded_citations`,
the `rewritten_query`, `retrieval_mode`, `prompt_version`, `model`,
`tokens_in/out`, `cost_usd`, and `latency_ms`.

### 5. API ([`app/api.py`](app/api.py)) & UI ([`app/ui.py`](app/ui.py))

- `POST /ask` → full `Answer` JSON (+ an `answer_id`); logs the conversation.
- `POST /ask/stream` → server-sent events: `sources`, then `token` deltas, then
  a `done` event with the full answer — or an `error` event if generation fails
  mid-stream. Upstream retrieval/model failures on `/ask` return a clean `502`
  (details logged server-side, not leaked to the client).
- `POST /feedback` → records 👍/👎 (+ optional comment) for a prior `answer_id`.
- `GET /health` → checks Qdrant + Postgres connectivity.

Conversation logging is **best-effort** — a monitoring outage never blocks an
answer. The Streamlit UI is a single page: question box, document filter,
streaming cited answer, source snippets (the ones the answer actually cited are
marked and expanded first), feedback buttons, follow-ups,
and a sidebar showing the last answer's model / cost / latency.

## Project structure

```
irb-copilot/
├── ingestion/            # the pipeline
│   ├── flow.py           #   Prefect orchestration (make ingest)
│   ├── pipeline.py       #   the 4 stages as reusable functions
│   ├── download.py  parse.py  chunk.py  index.py
│   ├── models.py         #   Paragraph / ParsedDoc / Chunk + deterministic ids
│   └── tokens.py         #   tiktoken helpers, text normalization
├── app/                  # the application
│   ├── config.py         #   pydantic-settings; single source of config
│   ├── providers.py      #   OpenAI client, chat + streaming + cost accounting
│   ├── retrieval.py      #   4 retrieval modes + RRF
│   ├── rewrite.py  prompts.py
│   ├── rag.py            #   answer() / answer_stream() orchestration
│   ├── api.py            #   FastAPI
│   └── ui.py             #   Streamlit
├── evaluation/           # ground truth + retrieval/RAG evaluation
│   ├── generate_ground_truth.py   metrics.py   sampling.py   corpus.py
│   ├── eval_retrieval.py   eval_rag.py   judge.py
│   └── results/          #   committed CSVs + PNG plots
├── monitoring/
│   ├── db.py  schema.sql            #   conversations + feedback
│   └── grafana/provisioning/        #   datasource + 6-panel dashboard
├── deploy/               # VM deployment (compose overlay, Caddy, scripts)
├── data/sources.yaml     # corpus registry (URLs + sha256); PDFs gitignored
├── notebooks/experiments.ipynb
├── docker-compose.yml   Dockerfile   Makefile   pyproject.toml   uv.lock
└── tests/                # 84 tests
```

## Dataset

Seven public regulatory PDFs, registered in
[`data/sources.yaml`](data/sources.yaml) with official URLs + pinned SHA-256. PDFs
are **not** committed; the pipeline downloads and verifies them.

| id | Document |
|----|----------|
| `ebagl_2017_16` | EBA Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures |
| `ebagl_2020_06` | EBA Guidelines on loan origination and monitoring |
| `ebagl_2016_07` | EBA Guidelines on the application of the definition of default (Art. 178 CRR) |
| `ecb_gim_2024` | ECB Guide to Internal Models (Feb 2024, consolidated) |
| `bcbs_d424_irb` | Basel III: Finalising post-crisis reforms (IRB chapters) |
| `ebagl_2019_03` | EBA Guidelines on downturn LGD estimation |
| `ebagl_2020_05` | EBA Guidelines on credit risk mitigation for A-IRB institutions |

docling parsing yields **2,968 numbered paragraphs → 2,241 structure-aware chunks**
across the seven documents.

## Run it yourself (reviewer walkthrough)

Verified end to end from a fresh clone of this repository. Each step says what
you should see, so you can tell a slow step from a stuck one.

**Before you start**

| | |
|---|---|
| Docker Desktop | running (Qdrant, Postgres, Grafana) |
| An OpenAI API key | ~**$0.05** for ingestion + a few questions |
| Disk | ~3 GB (PDFs, Python deps, docling models) |
| Time | ~10 min, most of it the first `make ingest` |

`uv` is the only tool you may need to install:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

---

**1 — Install and configure** (~1 min, or ~3 with a cold `uv` cache)

```bash
make setup            # uv sync from the lockfile, then copies .env.example -> .env
```

Now open `.env` and set `OPENAI_API_KEY=sk-...`. Leave everything else alone —
the defaults are the configurations the evaluation selected.

```bash
uv run pytest -q      # 312 passed  <- confirms the checkout is sound before you spend anything
```

**2 — Start the backing services** (~30 s)

```bash
make up               # qdrant + postgres + grafana
docker compose ps     # all three should read "running"
```

**3 — Build the corpus** (~6 min, ~$0.02)

```bash
make ingest
```

This downloads seven official PDFs (sha256-verified), parses them with docling,
chunks them, and indexes into Qdrant + BM25. You should see, per document:

```
[parse] ebagl_2017_16: 368 paragraphs -> .../data/parsed/ebagl_2017_16.json
[chunk] ebagl_2017_16: 337 structure chunks -> ...
[index] 2241 structure chunks from 7 doc(s) -> collection=irb_chunks, bm25=bm25_index
```

**2,968 paragraphs → 2,241 chunks, and no `[parse] WARNING` lines.** A warning
would mean paragraph detection failed and text was absorbed under the wrong
citation anchor — the defect [described below](#what-counts-as-a-hit).

> The first run downloads ~2 GB of docling layout models from HuggingFace. If it
> appears to hang here with no output, that is the Hub being rate-limited, not
> the pipeline: `hf auth login` with a free token, or retry later.

**4 — Run it** (~10 s)

```bash
make run              # FastAPI on :8000, Streamlit on :8501
```

Open **http://localhost:8501** and ask:

> *How many days past due trigger a default?*

**What to look for** — this is the part worth your attention:

- The answer streams in with an inline citation after each claim.
- Under **Sources**, the ones the answer actually cited are marked **✓ and opened
  first**; uncited retrievals stay collapsed. Open a cited source and check the
  paragraph really says what the answer claims. That verification loop is the
  whole point of the tool.
- Each source shows its **retrieval rank** and score. Rank is the meaningful
  signal — scores are not comparable across retrieval modes.
- Ask a follow-up like *"and what is the materiality threshold for that?"* A
  caption then shows **the query retrieval actually ran on**, which is a rewritten
  standalone version of your question, not the words you typed.
- Try something outside the corpus (*"what is the capital of France?"*). The
  answer should decline rather than guess, and a banner notes that it cited
  nothing.

Then **http://localhost:3000** (`admin`/`admin`) for the Grafana dashboard. Your
questions appear under `source = 'live'`; the panels deliberately exclude the
evaluation's own synthetic traffic.

---

**Reproducing the evaluation** is optional and slower — roughly 3 hours and ~$4,
because it re-runs 24 retrieval configurations over 796 questions twice, plus a
100-question RAG grid:

```bash
make eval-retrieval        # standard ground truth
make eval-retrieval-hard   # de-biased ground truth  <- this one picks the defaults
make eval-rag              # prompt x model grid, LLM-as-a-judge
```

The committed results in [`evaluation/results/`](evaluation/results/) are what the
tables below quote, and `uv run pytest tests/test_readme_claims.py` checks that
every number in this file still matches those CSVs.

**Troubleshooting**

| Symptom | Cause |
|---|---|
| `address already in use` on `make run` | the Docker `api`/`ui` containers hold 8000/8501 — `docker compose stop api ui` |
| `/ask` returns 401 | `API_KEY` is set in your `.env`; leave it empty for local use |
| `429 ... exceeded your current quota` | the OpenAI account is out of credit (this is billing, not rate limiting) |
| `no chunk files ... run ingestion first` | step 3 has not completed |

## Running the app

Two ways to run it — **don't run both at once**, they both bind ports 8000/8501:

| Mode | Commands | Use when |
|------|----------|----------|
| **Local dev** (hot-reload) | `make up` (deps only) → `make run` | developing; fastest feedback |
| **All in Docker** | `make up-all` (or `docker compose up -d --build`) | run everything containerized |

> Port clash? If `make run` says *"address already in use"*, the Docker `api`/`ui`
> containers are holding the ports — `docker compose stop api ui`, then `make run`.
> (`make up` starts only qdrant + postgres + grafana; `make up-all` adds api + ui.)

Then open:
- **UI** → http://localhost:8501 — type a question, optionally restrict to specific
  documents; the cited answer **streams in**. Sources the answer actually cited
  are marked ✓ and opened first — verifying a claim is the real task — with the
  rest collapsed underneath. Warns when an answer cites nothing, is cut off at
  the token cap, or cites something the retrieved sources do not support. 👍/👎
  feedback and follow-up questions (a short conversation history is kept).
- **Grafana** → http://localhost:3000 (`admin` / `admin`) — the monitoring dashboard.

### API

```bash
# Ask (full JSON answer with citations, cost, latency)
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question":"How many days past due trigger a default?"}' | jq .

# Streaming (server-sent events: sources → tokens → final answer)
curl -N localhost:8000/ask/stream -H 'Content-Type: application/json' \
  -d '{"question":"What is a technical default?"}'

# Feedback on a prior answer, and health
curl -s localhost:8000/feedback -H 'Content-Type: application/json' \
  -d '{"answer_id":"<id from /ask>","thumbs":"up"}'
curl -s localhost:8000/health          # {"status":"ok","qdrant":true,"postgres":true}
```

Restrict retrieval to specific documents by passing `"doc_ids": ["ebagl_2016_07"]`.
The image is multi-stage (uv, non-root) with **CPU-only torch**; ingest inside the
Docker stack with `docker compose exec api python -m ingestion.flow`.

## Configuration

All behaviour is env-driven via [`.env.example`](.env.example) → `.env`
([`app/config.py`](app/config.py) is the single source of truth). Defaults encode
the evaluation winners, so the app is sensible out of the box.

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_API_KEY` | — | OpenAI key (required at runtime; empty is fine for tests) |
| `OPENAI_BASE_URL` | — | point at an OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | chat model for answers/rewrite |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `text-embedding-3-small` / `1536` | embeddings (dim must match) |
| `EVAL_MODELS` | `gpt-4o-mini,gpt-4o` | models compared by `eval_rag` |
| `RETRIEVAL_MODE` | `hybrid_rerank` | `bm25` \| `vector` \| `hybrid` \| `hybrid_rerank` |
| `REWRITE_MODE` | `off` | first-turn query rewriting: `off` \| `glossary` (acronym expansion) \| `llm` |
| `CONDENSE_HISTORY` | `true` | rewrite follow-ups into standalone retrieval queries |
| `PROMPT_VERSION` | `v2` | `v1` (plain) \| `v2` (few-shot, stricter citations) |
| `CHUNKER` | `structure` | `structure` \| `naive` (eval baseline) |
| `TOP_K` / `RRF_K` | `5` / `60` | results returned / RRF constant |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder for `hybrid_rerank` |
| `QDRANT_URL` / `QDRANT_COLLECTION` | `http://localhost:6333` / `irb_chunks` | vector store |
| `POSTGRES_DSN` | `postgresql+psycopg://irb:irb@localhost:5432/irb` | monitoring DB |
| `API_PORT` / `UI_PORT` / `GRAFANA_PORT` | `8000` / `8501` / `3000` | service ports |
| `API_URL` | `http://localhost:8000` | where the UI calls the API |
| `API_KEY` | — | if set, `/ask` `/ask/stream` `/feedback` require an `X-API-Key` header |
| `RATE_LIMIT_PER_MINUTE` | `30` | per-client-IP request limit (0 disables) |
| `TRUSTED_PROXY_HOPS` | `1` | reverse proxies in front (Caddy = 1); real client IP is read this many hops from the right of `X-Forwarded-For`. Set `0` when no proxy |
| `MAX_QUESTION_CHARS` / `MAX_HISTORY_MESSAGES` / `MAX_DOC_IDS` | `2000` / `10` / `20` | request input bounds |
| `BIND_HOST` / `SITE_ADDRESS` | `0.0.0.0` / `:80` | production deploy (see [deploy/](deploy/README.md)) |

## Evaluation

Reproducible; artifacts committed under
[`evaluation/results/`](evaluation/results/). The ground truth is LLM-generated
questions whose answer sits in a sampled chunk (stratified by document).

### How this evaluation was built (and what it took to trust it)

Measuring a RAG system is mostly a fight against measuring the wrong thing. Each
subsection below is a mistake that was actually made here, what it did to the
numbers, and how it was fixed — the fixes are the reason the tables further down
are worth reading.

**1. LLM-generated ground truth is biased toward lexical search.**
The ground truth is built by sampling chunks and asking a model to write
questions answerable from each. Those questions reuse the passage's own
vocabulary, so BM25 can win by matching words rather than meaning. That bias is
measurable: the **mean question↔chunk lexical overlap is 0.78**. The fix is a
second ground truth (`--style hard`) that instructs the generator to paraphrase
heavily, dropping overlap to **0.43**. Neither set is "correct" — the *gap
between them* is the finding, and it is what separates real retrieval quality
from vocabulary matching.

**2. A loose relevance rule inflates every score, unevenly.**
See [What counts as a hit](#what-counts-as-a-hit). Matching a retrieved chunk to
the ground truth by paragraph number seems natural and is wrong here, because
these regulations restart numbering in every section. It accepted **6.7 chunks
per question** where it should accept ~1 — and it flattered the fixed-window
chunker more than the structure-aware one, because a larger window sweeps up more
recurring numbers. A metric that is generous is bad; a metric that is generous
*unevenly across the things being compared* invalidates the comparison.

**3. An ablation must run the production code path, not a reimplementation.**
The "no rewriting" arm originally retrieved on the raw question, while the app
itself always applied acronym expansion — so the shipped configuration was never
the configuration measured. Every arm now calls the same `rewrite_query` the
application calls, differing only by setting. Fixing that exposed a second
problem: acronym expansion had *never* been evaluated at all, because the flag
that supposedly disabled rewriting only disabled the LLM half of it.

**4. Aggregate effects hide where a feature only fires on a subset.**
Acronym expansion looked like noise across all questions (~1pt). But only 25% of
questions contain an acronym, and just **3%** of the de-biased set — for the rest,
"expansion on" and "expansion off" are literally the same query string. Measured
only on the questions it can affect, it clearly hurts: **−0.035** hit@5 on BM25,
**−0.050** on vector. Any feature that fires conditionally needs a conditional
measurement, or the aggregate will average the effect away.

**5. A judge that cannot be read is missing data, not a bad answer.**
Malformed or truncated judge output was being scored `NON_RELEVANT`, which is
indistinguishable from a genuine negative verdict — and it penalised whichever
configuration made the judge most verbose, which looks exactly like a quality
difference. Unreadable verdicts are now a distinct `PARSE_ERROR`, retried once,
and excluded from the denominator. The CSV reports `n → answered → judged` so a
shrunken sample is always visible rather than implied.

**6. The judge must not share a family with what it judges.**
Models prefer their own output. `JUDGE_MODEL` is `gpt-5.4-mini` while answers come
from `gpt-4o-mini`, and every row carries a `self_judged` flag computed on the
model *family* — `gpt-4o` and `gpt-4o-mini` are the same family, so name equality
is not enough.

**7. Report noise before reporting rankings.**
Three configurations were measured twice, giving a repeat-measurement noise floor
of ~2–3pt on relevance and ~3–6pt on citation support. That single number changes
what may be claimed: the prompt effect (+14pt citations) is real, while the
model effect (+4pt relevance) sits close enough to the noise that it is reported
as a cost trade rather than a win.

**8. What survives a rebuild is what you can believe.**
After the parser was fixed, the corpus was rebuilt from the PDFs — 38% more
chunks — and everything re-run. `structure > naive` (12/12) and `glossary > llm`
(8/8) held on both ground truths; the rank ordering and one prompt/model
conclusion changed. Conclusions that survive a corpus rebuild are findings;
conclusions that do not were artifacts.

### What counts as a hit

Before the numbers, the relevance rule, because it decides what they mean. Each
ground-truth question was written from one specific chunk, so a retrieved chunk
counts as relevant only if it **is** that passage: an exact chunk-id match, or it
reproduces ≥30% of the passage's word 5-grams (so a differently-merged chunk that
still contains the answer counts). The same rule applies to both chunkers, which
is what makes the structure-vs-naive comparison fair.

Matching on document + paragraph number instead — the obvious shortcut — does not
work here: these regulations restart paragraph numbering in every section, so
"paragraph 82" is dozens of different passages. On this corpus that rule admits a
mean of **2.4 chunks per question** (worst case 22), against **1.4** for the text
rule. It was far worse before the parser was fixed to stop annexes accumulating
under the last numbered paragraph: 6.7 on average, worst case 51.

Two limitations worth stating. Twelve of the 796 questions (1.5%) cannot be
matched to *any* fixed-window chunk at any threshold, so they are guaranteed
misses for that chunker — a small floor bias against the naive baseline, which
already loses every comparison, so it cannot explain the result. And a handful of
ground-truth questions were generated from short, repeated table headers, which
match many chunks; those inflate rather than depress the scores.

### Retrieval (`eval_retrieval.py`) — hit-rate@5 / MRR@5

24 configs = {bm25, vector, hybrid, hybrid_rerank} × {structure, naive} ×
{`off`, `glossary`, `llm`} rewriting, over 796 questions, on two ground truths
([standard CSV](evaluation/results/retrieval_eval.csv) ·
[de-biased CSV](evaluation/results/retrieval_eval_hard.csv)).

LLM-generated questions reuse the source's vocabulary — mean question↔chunk
lexical overlap **0.78** — which flatters lexical search. A **de-biased** ground
truth (`--style hard`) paraphrases them (overlap **0.43**), and comparing the two
separates genuine retrieval quality from that artifact (structure chunks, no
rewriting):

| mode | standard hit@5 | **de-biased** hit@5 | change |
|------|:---:|:---:|:---:|
| bm25 | 0.899 | 0.545 | **−0.354** |
| vector | 0.766 | 0.637 | −0.129 |
| hybrid | 0.877 | 0.694 | −0.183 |
| **hybrid_rerank** | **0.902** | **0.716** | −0.186 |

- standard order: `hybrid_rerank > bm25 > hybrid > vector`
- de-biased order: `hybrid_rerank > hybrid > vector > bm25`

**BM25 falls from second to last.** It is within a third of a point of the best
config when questions echo the passage, and 17pt behind it when they don't —
losing **35 points** between the two sets, against 19 for `hybrid_rerank`. Pure
vector search is the most *robust* to paraphrase (−0.129) without ever being the
best; the cross-encoder is what converts that robustness into the top score.

**Chosen default: `hybrid_rerank` + `structure` + `off`** — it wins **both** ground
truths (standard hit@5 **0.902** / MRR@5 **0.7947**; de-biased **0.7161** /
**0.5571**), so the choice does not depend on trusting the de-biased set over the
standard one. It adds an embedding lookup and a cross-encoder pass, so it is
slower than BM25: a deliberate quality-for-latency trade. Set
`RETRIEVAL_MODE=bm25` for the fast lexical path.

Two further results:

- **Query rewriting never helps.** The LLM reformulation is clearly harmful —
  worse than `glossary` in **8/8** mode×chunker combinations on *both* ground
  truths, by 15–20pt. Deterministic acronym expansion is milder: consistently
  negative on the standard set (8/8) but indistinguishable from zero on the
  de-biased one (6/8, margins of 0.3pt), because only 3% of those questions
  contain an acronym at all — for the rest, `off` and `glossary` are literally the
  same query. Both levels move the query away from the exact regulatory wording
  the documents use, which is why `REWRITE_MODE=off` ships — see
  [the ablation note](#query-rewriting-ablation).
- **Structure-aware chunking beats the fixed-window baseline in 12/12**
  mode×rewrite comparisons, on *both* ground truths, and it held after the corpus
  was re-parsed from scratch with 38% more chunks. Re-ranking partly compensates
  for bad chunking: on the standard set it lifts naive chunks from 0.780 (hybrid)
  to 0.857, a reasonable argument for keeping a re-ranker as insurance.

Reproduce with `make eval-retrieval` and `make eval-retrieval-hard`. Only the
de-biased run should set defaults; the tool says so explicitly when it finishes.

<a id="query-rewriting-ablation"></a>
#### Why acronym expansion was dropped

`glossary` mode appends `LGD (loss given default)` to the query. Aggregated over
all questions its effect looks like noise (≈1pt), because most questions contain
no acronym at all — 25% on the standard set and just **3%** on the de-biased set.
Measured only on the questions where it actually changes the query (n=200), it
clearly hurts: **−0.035** hit@5 on bm25 and **−0.050** on vector. The expansion
dilutes term weights and pulls the embedding away from the corpus's own
vocabulary, which uses the acronyms too.

This is why the de-biased set shows `glossary` beating `off` in two of eight
configs by ~0.3pt: with 97% of its questions unaffected, that arm has almost no
signal to carry a sign. Reporting it as a tie would be more honest than reading a
direction into it — the effect is only measurable where acronyms occur.

Caveat worth stating: ground-truth questions are generated *from* the passages, so
they inherit the corpus's acronym usage. A user typing "MoC" against a document
that spells out "margin of conservatism" is the case where expansion *should*
help, and this ground truth cannot represent it. The measured harm is real; a
possible benefit is under-measured. Revisitable against real query logs, which the
monitoring DB collects.

### RAG (`eval_rag.py`) — LLM-as-a-judge

For each (prompt version × answer model) config, a judge model labels every answer
RELEVANT / PARTLY_RELEVANT / NON_RELEVANT against the question and the ground-truth
passage, and *separately* checks whether each `[doc, para. X]` citation is
supported by the sources the answer was shown. Cost and latency are recorded per
config ([CSV](evaluation/results/rag_eval.csv)).

Three things the harness does deliberately, because each is a way this kind of
evaluation usually goes wrong:

- **A cross-family judge.** `JUDGE_MODEL` defaults to `gpt-5.4-mini` while answers
  come from `gpt-4o-mini`. A judge scoring its own family shows self-preference
  bias, so every row carries a `self_judged` flag — computed on the model *family*
  (`gpt-4o-mini` and `gpt-4o` are the same family), not on the exact name — and
  the run warns loudly when it is set.
- **Unreadable verdicts are missing data, not bad answers.** A malformed or
  truncated judge response becomes `PARSE_ERROR`, is excluded from the rate
  denominators, and is reported in its own `parse_errors` column and as a grey
  segment on the plot. Folding those into NON_RELEVANT (the obvious shortcut)
  understates every config, and worst the one whose answers make the judge most
  verbose — a bias that looks exactly like a quality difference.
- **Rates are over answers actually judged** (`judged` column = n − parse errors);
  cost and latency stay over all n, because they were really spent.

The run also seeds the monitoring DB with judged conversations, so the Grafana
judge-relevance panel has data.

**100 questions × 4 configs** = {`gpt-4o-mini`, `gpt-4o`} × {`v1`, `v2`}. Every
config answered and judged all 100, with zero parse errors:

| model | prompt | RELEVANT | cite-ok | cost/q |
|-------|--------|:---:|:---:|:---:|
| gpt-4o-mini | v1 | 0.84 | 0.64 | $0.00032 |
| **gpt-4o-mini** | **v2** | 0.85 | **0.78** | **$0.00033** |
| gpt-4o | v1 | **0.89** | **0.78** | $0.00509 |
| gpt-4o | v2 | 0.88 | **0.78** | $0.00547 |

**Chosen default: `gpt-4o-mini` + prompt `v2`.** Note that this is *not* the
highest-relevance config — `gpt-4o` + `v1` scores 0.89 against 0.85. It is a
deliberate cost trade: **4pt of relevance for 15× the price per question**, with
**identical** citation support (0.78). Citation correctness is the metric this
domain actually cares about, and it is tied. Set `LLM_MODEL=gpt-4o` if you want
those 4 points.

**The prompt matters more than the model — but read the size of each effect.**
Three configs were measured twice across runs, giving a repeat-measurement noise
floor of ~2–3pt on relevance and ~3–6pt on citation support.

- **`v1` → `v2` on gpt-4o-mini: +14pt citation support** (0.64 → 0.78), far
  outside the noise and consistent across both runs. The relevance gain is only
  +1pt. The stricter few-shot prompt does specifically what it was written to
  do — enforce citations — and little else.
- **`v2` does nothing for gpt-4o** (0.89 → 0.88 relevance, 0.78 → 0.78
  citations). Few-shot scaffolding lifts the weaker model to the stronger one's
  citation discipline and is redundant once the model already has it. A prompt
  improvement should not be assumed to transfer across models.
- **gpt-4o's relevance edge is real but small** (+4pt, marginally outside noise),
  and it buys nothing on citations. In an earlier run on the previous corpus the
  comparison went the other way (gpt-4o-mini ahead by 1pt), so treat 4pt as the
  upper end of what the larger model is worth here.

> **Latency is deliberately not reported.** The harness measures end-to-end wall
> time, and a few requests stall for minutes under concurrency — one outlier of
> 606 s moved a config's *mean* from ~3 s to ~18 s, which would have made the
> cheap model look 7× slower than the expensive one. Medians are comparable
> (3.3 s vs 2.0 s). The harness now records `p50_latency_ms` and
> `p95_latency_ms`; the committed CSV predates that column.

Throughput, unlike per-request latency, does differ sharply: gpt-4o's 30k TPM
ceiling on this account sustains only ~8 requests/minute against ~80 for
gpt-4o-mini, which is why the evaluation paces it to one request in flight (see
`MODEL_MAX_CONCURRENCY`).

**Prompt versions.** `v1` states the grounding and citation rules plainly; `v2`
adds a worked few-shot example and requires a citation on every factual sentence.
`PROMPT_VERSION=v2` ships. Both are in [`app/prompts.py`](app/prompts.py).
Reproduce with `make eval-rag`.

Both evals run their hundreds of API calls concurrently (`--workers`, default 8).

## Monitoring

Every `/ask` is logged to Postgres (`conversations`: question, rewritten query,
answer, model, prompt/retrieval mode, chunk ids, tokens, cost, latency,
`judge_relevance`, whether the answer was cut off at the token cap
(`answer_truncated`), and the citation self-check result `citations_grounded` /
`ungrounded_citations`), and every 👍/👎 to `feedback`. Grafana is
auto-provisioned (datasource + dashboard) with eight panels:

1. questions per day · 2. thumbs up/down ratio · 3. cost over time ·
4. latency p50/p95 · 5. retrieval-mode usage · 6. judge-relevance distribution ·
7. ungrounded-citation rate (hallucination trend) · 8. truncated-answer rate.

**Real traffic and evaluation traffic are separated.** `eval_rag` seeds hundreds
of judged conversations so panel 6 has data — and until a `source` column was
added they were indistinguishable from real usage: **1,901 of 1,921 rows** were
the evaluation's own activity, so every usage, cost and latency panel was
describing the harness rather than users. Panels 1–5, 7 and 8 now filter to
`source = 'live'`; panel 6 reads `source = 'eval'`. No panel mixes the two, and a
test enforces that every panel states its scope.

Schema in [`monitoring/schema.sql`](monitoring/schema.sql); it's applied on first
Postgres init and idempotently by `monitoring.db` at API startup, which also adds
the newer columns to an existing database.

## Screenshots

| Streamlit UI | Grafana dashboard |
|--------------|-------------------|
| ![UI](docs/screenshots/ui.png) | ![Grafana](docs/screenshots/grafana.png) |

> To capture: run `make run`, ask a question, screenshot the answer + sources →
> `docs/screenshots/ui.png`; open Grafana → "IRB Copilot — Monitoring" →
> `docs/screenshots/grafana.png`.

## Deployment

Deploy the whole stack to a single VM with docker-compose behind a **Caddy**
reverse proxy (automatic HTTPS; only 80/443 exposed — Qdrant/Postgres/Grafana
stay on the internal network). Full runbook in
**[deploy/README.md](deploy/README.md)**.

```bash
sudo bash deploy/provision.sh   # install Docker + firewall (once)
cp .env.example .env            # set OPENAI_API_KEY, SITE_ADDRESS, BIND_HOST=127.0.0.1
bash deploy/deploy.sh           # build, start, ingest
# -> UI at https://<domain>/, API at /api, Grafana at /grafana
```

## Security

Because `/ask` fans out to paid OpenAI calls (and a CPU reranker), a publicly
reachable API is a cost/DoS surface. The write endpoints (`/ask`, `/ask/stream`,
`/feedback`) are defended in layers:

- **Input bounds** (always on) — question length, history size, and `doc_ids`
  count are capped (`MAX_*` settings) so a single request can't blow up the
  prompt; oversized requests get a `422`.
- **Prompt-injection defense** — conversation `history` is schema-validated to
  `user`/`assistant` roles only (a client can't inject a `system` message → `422`),
  and `build_messages` re-sanitizes it as a safety net regardless of caller.
- **Per-IP rate limiting** — `RATE_LIMIT_PER_MINUTE` (default 30) returns `429`
  when exceeded. The client IP is read `TRUSTED_PROXY_HOPS` (default 1, for Caddy)
  hops from the **right** of `X-Forwarded-For` — the value the trusted proxy
  appended — so a client can't spoof the leftmost hop to rotate the limiter key.
- **Optional API key** — set `API_KEY` and the endpoints require an `X-API-Key`
  header (the UI sends it automatically); leave empty for an open demo.
- **Edge body-size cap** — Caddy rejects request bodies over 64 KB before they
  reach the app.

For a public deployment, set `API_KEY` and a conservative `RATE_LIMIT_PER_MINUTE`,
and change `GRAFANA_PASSWORD` / `POSTGRES_PASSWORD` from their defaults. The
limiter is in-memory per process — for multiple workers/instances, back it with
Redis.

## Development

```bash
make test     # pytest (84 tests: chunker, parser, retrieval, RAG, API, eval, pipeline)
make lint     # ruff
make fmt      # ruff format
```

Conventions: type hints on public functions; pydantic models at boundaries
(API, config); no hard-coded secrets or model names (everything via `config.py`);
fail loudly (no bare `except`, no silent skips); every LLM call goes through
`providers.py` and is returned with token counts + cost. Tests avoid external
services by injecting counters, using fake docling items, and mocking the RAG/DB
layers, so the suite runs offline in ~1 s.

## Design decisions & trade-offs

- **Deterministic chunk ids include an ordinal.** Regulatory docs restart
  paragraph numbering per section, so `hash(doc_id + para_ids)` collides; the
  ordinal makes ids unique yet stable across re-runs (upsert, don't duplicate).
- **docling + reranker are default dependency *groups*, not extras.** `uv run`
  re-syncs to the default set and would uninstall optional extras every
  invocation; groups (installed by default) avoid that footgun.
- **CPU-only torch in Docker.** docling/sentence-transformers pull CUDA torch
  (~2 GB of nvidia packages) by default; the lockfile pins `torch+cpu` on Linux,
  halving the image.
- **`hybrid_rerank` as the default.** It wins *both* ground truths, so the
  choice does not rest on trusting the de-biased set over the standard one. What
  the two sets reveal is how badly lexical search degrades: BM25 loses 35pt
  between them and falls from second place to last, against 19pt for the winner.
  Latency is the trade (set `bm25` for speed).
- **Config paths are anchored to the repo root**, so the app works from any cwd
  (CLI, notebook, container, API).
- **`Answer` is a pydantic model** (a boundary type serialized by the API and
  consumed by the UI), superseding the plain dataclass in the original spec.
- **Unnumbered content gets an empty citation anchor, not a fabricated one.**
  Annexes and tables carry no paragraph number; inventing one is what previously
  filed 48% of the corpus under the wrong citation. They now cite document +
  section, and 24% of chunks are honestly unnumbered.
- **Relevance is decided by text, not by paragraph number** — one rule for both
  chunkers, which is also what makes the structure-vs-naive comparison fair.

## Rubric mapping

| Criterion | Where |
|-----------|-------|
| Problem described | [Problem](#problem) |
| RAG flow: knowledge base + LLM | `app/rag.py` (retrieve → prompt → LLM) |
| Retrieval: multiple approaches evaluated, best used | `app/retrieval.py` (4 modes), `evaluation/eval_retrieval.py`, default = de-biased winner |
| LLM: multiple prompts/models evaluated, best used | `app/prompts.py` (v1/v2), `evaluation/eval_rag.py`, default = winner |
| Interface: UI + API | `app/ui.py` (Streamlit) + `app/api.py` (FastAPI, incl. streaming) |
| Ingestion: automated pipeline (special tool) | **Prefect** flow `ingestion/flow.py` (`make ingest`); stages in `ingestion/pipeline.py` |
| Monitoring: feedback + dashboard (5+ panels) | `/feedback`, `monitoring/db.py`, Grafana (8 panels), with evaluation traffic excluded from the usage panels |
| Containerization | full `docker-compose.yml` + multi-stage `Dockerfile` (uv, non-root, CPU-only torch); build verified |
| Deployment | VM deploy via `deploy/` — production overlay behind a Caddy TLS proxy, [runbook](deploy/README.md) |
| Reproducibility | `make setup`, auto-download + sha256, `uv.lock` pins all versions, committed eval results. Verified from a fresh clone: the rebuilt corpus reproduces a **byte-identical fingerprint**, so all 796 ground-truth rows and every committed result resolve against a corpus you build yourself |
| Best practices: hybrid search / re-ranking / query rewriting | all implemented (`retrieval.py`, `rewrite.py`) and evaluated |

See [SPEC.md](SPEC.md) for the full original specification.
