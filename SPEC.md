# IRB Copilot — Project Specification

> Implementation spec for Claude Code. Read this file fully before writing any code.
> Target: final project for the DataTalks.Club LLM Zoomcamp (peer-reviewed against a public rubric — see §12).

## 1. Purpose

Build **IRB Copilot**, a RAG-based Q&A assistant over public EU prudential regulation for credit risk modeling (IRB / definition of default / loan origination). Users are credit risk analysts who ask questions like:

- "What does the EBA require regarding margin of conservatism in LGD estimation?"
- "How many days past due trigger default under the definition of default guidelines?"
- "What does the ECB guide say about representativeness of reference datasets?"

Every answer MUST cite its sources with document + paragraph references (e.g., `EBA/GL/2017/16, para. 82`). Uncited answers are unacceptable in this domain.

## 2. Tech stack (non-negotiable)

- **Python 3.12**, managed with **uv** (`pyproject.toml` + `uv.lock`). No pip, no poetry, no conda.
- **Qdrant** as vector store (Docker service).
- **PostgreSQL** for monitoring/feedback logs (Docker service).
- **Grafana** for the monitoring dashboard (Docker service, provisioned from JSON).
- **FastAPI** backend, **Streamlit** front end (separate processes, Streamlit calls the API).
- **OpenAI API** for LLM + embeddings by default, but ALL model access goes through a thin provider module so models can be swapped via env vars.
- **docling** for PDF parsing (fallback: pymupdf if docling fails on a document).
- **pytest** for tests. Type hints everywhere. `ruff` for lint/format.

## 3. Repository layout

```
irb-copilot/
├── README.md
├── SPEC.md                    # this file
├── pyproject.toml
├── uv.lock
├── docker-compose.yml         # qdrant + postgres + grafana + api + ui
├── Dockerfile                 # multi-stage, runs API by default
├── .env.example
├── Makefile                   # make ingest / eval / run / test
├── data/
│   ├── sources.yaml           # document registry: id, title, official URL, sha256
│   └── raw/                   # downloaded PDFs (gitignored)
├── ingestion/
│   ├── __main__.py            # `python -m ingestion` runs the full pipeline
│   ├── download.py
│   ├── parse.py
│   ├── chunk.py
│   └── index.py
├── app/
│   ├── config.py              # pydantic-settings, reads .env
│   ├── providers.py           # LLM + embedding client factory
│   ├── retrieval.py           # bm25 | vector | hybrid (+ optional rerank)
│   ├── rewrite.py             # acronym expansion / query rewriting
│   ├── prompts.py
│   ├── rag.py                 # orchestrates rewrite → retrieve → prompt → LLM
│   ├── api.py                 # FastAPI app
│   └── ui.py                  # Streamlit app
├── evaluation/
│   ├── generate_ground_truth.py
│   ├── eval_retrieval.py
│   ├── eval_rag.py
│   └── results/               # committed CSVs + PNG plots referenced by README
├── monitoring/
│   ├── db.py                  # SQLAlchemy models + logging helpers
│   ├── schema.sql
│   └── grafana/               # provisioning: datasource + dashboard JSON
├── tests/
│   ├── test_chunk.py
│   ├── test_retrieval.py
│   └── test_api.py
└── notebooks/
    └── experiments.ipynb
```

## 4. Corpus

`data/sources.yaml` registers these documents (find current official PDF URLs from eba.europa.eu, bankingsupervision.europa.eu, bis.org; verify each downloads correctly):

| id | Document |
|----|----------|
| `ebagl_2017_16` | EBA Guidelines on PD estimation, LGD estimation and treatment of defaulted exposures |
| `ebagl_2020_06` | EBA Guidelines on loan origination and monitoring |
| `ebagl_2016_07` | EBA Guidelines on the application of the definition of default (Art. 178 CRR) |
| `ecb_gim_2024` | ECB Guide to Internal Models (consolidated) |
| `bcbs_d424_irb` | Basel III: Finalising post-crisis reforms (IRB chapters only) |

Rules:
- `download.py` fetches each PDF, verifies sha256 against `sources.yaml` (populate hashes on first successful download), and is idempotent.
- If a URL 404s, fail loudly with the document id — do NOT silently skip.
- Do not commit PDFs to git.

## 5. Ingestion pipeline (`python -m ingestion`)

Stages, each skippable/resumable via CLI flags (`--from-stage parse`, `--only-doc ebagl_2017_16`):

1. **download** — as above.
2. **parse** (`parse.py`) — docling → structured output. Extract: section hierarchy, numbered paragraphs, page numbers. Regulatory documents number their paragraphs explicitly ("82.", "Article 178(1)(b)") — capture these numbers; they are the citation anchors. Strip headers/footers/TOC. Tables: render to markdown, keep attached to their paragraph.
3. **chunk** (`chunk.py`) — **structure-aware chunking**: one chunk per numbered paragraph; merge consecutive paragraphs under the same subsection if combined length < 250 tokens; hard-split any chunk > 1000 tokens at sentence boundaries. Each chunk carries metadata:
   ```json
   {"doc_id": "...", "doc_title": "...", "section_path": ["Chapter 5", "5.3.2 Margin of conservatism"],
    "para_ids": ["82", "83"], "pages": [45], "text": "..."}
   ```
   Also implement a **naive baseline chunker** (fixed 500 tokens, 50 overlap) behind a flag — needed for the evaluation comparison.
4. **index** (`index.py`) — embed and upsert into Qdrant collection `irb_chunks` (deterministic IDs = hash(doc_id + para_ids), so re-runs upsert instead of duplicating). Store full metadata as payload. Also build/persist a BM25 index over the same chunks (use `bm25s` or rank_bm25; persist to disk).

## 6. Retrieval (`retrieval.py`)

Implement four search modes selected by config/env (`RETRIEVAL_MODE`):

- `bm25` — lexical over chunk text.
- `vector` — Qdrant cosine similarity.
- `hybrid` — reciprocal rank fusion of both (k=60).
- `hybrid_rerank` — hybrid top-20 → cross-encoder rerank (`BAAI/bge-reranker-base` via sentence-transformers, CPU is fine) → top-5.

All modes: return top-k (default 5) chunks with scores + metadata. Support optional metadata filter `doc_ids: list[str]` (UI lets the user restrict to specific documents).

## 7. Query rewriting (`rewrite.py`)

Before retrieval, run a cheap LLM call that (a) expands domain acronyms (PD, LGD, EAD, MoC, DoD, CCF, CRR, RDS…) from a hardcoded glossary dict first, LLM only if unknown terms remain, and (b) reformulates conversational phrasing into a search query. Toggleable via `ENABLE_REWRITE` — must be off-switchable for the ablation in evaluation.

## 8. RAG flow (`rag.py`)

`answer(question, doc_ids=None) -> Answer`:
1. rewrite (if enabled) → 2. retrieve → 3. build prompt → 4. LLM → 5. return.

Prompt requirements (`prompts.py`):
- System prompt: role = assistant for credit risk regulation; answer ONLY from provided context; if context is insufficient, say so explicitly — never guess; ALWAYS cite as `[doc_title, para. X]` inline after each claim.
- Context block: chunks with their metadata rendered as citation headers.
- Keep TWO prompt variants (`v1` plain, `v2` with few-shot example + stricter citation format) selectable via config — needed for LLM evaluation.

`Answer` dataclass: `text`, `citations` (parsed from the response), `chunks_used`, `model`, `tokens_in/out`, `cost_usd`, `latency_ms`.

## 9. API (`api.py`)

- `POST /ask` → `{question, doc_ids?}` → full Answer JSON. Logs to monitoring DB.
- `POST /feedback` → `{answer_id, thumbs: "up"|"down", comment?}`.
- `GET /health` → checks Qdrant + Postgres connectivity.

## 10. UI (`ui.py`)

Streamlit, single page: question box, multiselect for document filter, answer rendered with citations as expandable source snippets (show the actual chunk text), 👍/👎 buttons wired to `/feedback`, sidebar showing model/cost/latency of last answer. Keep it clean; no chat history needed (single-turn is fine).

## 11. Evaluation

### 11.1 Ground truth (`generate_ground_truth.py`)
Sample ~200 chunks (stratified by document). For each, LLM generates 4 questions a risk modeler might ask whose answer is contained in the chunk. Output `evaluation/ground_truth.csv`: `question, doc_id, chunk_id`. Print 20 random samples to stdout for manual spot-checking; add `--seed` for reproducibility.

### 11.2 Retrieval evaluation (`eval_retrieval.py`)
For each config in {bm25, vector, hybrid, hybrid_rerank} × {structure-aware chunks, naive chunks} × {rewrite on/off}: compute **hit-rate@5** and **MRR@5** against ground truth. Output `results/retrieval_eval.csv` + a bar chart PNG. The best config becomes the documented default in `.env.example`.

### 11.3 RAG evaluation (`eval_rag.py`)
On a 100-question sample: generate answers with (prompt v1 vs v2) × (gpt-4o-mini vs gpt-4o, or whatever `EVAL_MODELS` lists). LLM-as-a-judge classifies each answer RELEVANT / PARTLY_RELEVANT / NON_RELEVANT given question + ground-truth chunk, and separately checks citation correctness (do cited paragraphs support the claims?). Also record cost + latency per config. Output `results/rag_eval.csv` + distribution plot.

## 12. Monitoring

Postgres tables (`schema.sql`): `conversations` (id, ts, question, rewritten_query, answer, model, prompt_version, retrieval_mode, chunk_ids, tokens_in, tokens_out, cost_usd, latency_ms, judge_relevance nullable) and `feedback` (id, conversation_id, thumbs, comment, ts).

Grafana dashboard (auto-provisioned) with ≥5 panels: questions per day, thumbs up/down ratio, cost over time, latency p50/p95, retrieval mode usage, judge relevance distribution.

## 13. Docker & config

- `docker-compose up` brings up qdrant, postgres, grafana, api, ui — fully working after `docker-compose exec api python -m ingestion`.
- `Dockerfile`: multi-stage, uv-based install, non-root user.
- `.env.example` documents every variable: `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `RETRIEVAL_MODE`, `ENABLE_REWRITE`, `PROMPT_VERSION`, `QDRANT_URL`, `POSTGRES_DSN`, ports.
- `Makefile` targets: `setup`, `ingest`, `eval-retrieval`, `eval-rag`, `run`, `test`, `lint`.

## 14. README requirements

Written for a reviewer who did NOT take the course: problem statement, architecture diagram (mermaid), dataset description, **evaluation results tables copied from `results/`**, screenshots of UI + Grafana, exact run instructions from clone to first answer, and an explicit section mapping each rubric criterion to where it's satisfied in the repo.

## 15. Acceptance criteria (course rubric — maximize every line)

- [ ] Problem clearly described in README
- [ ] RAG flow: knowledge base + LLM both used
- [ ] Retrieval: multiple approaches evaluated, best one used in the app
- [ ] LLM: multiple prompts/models evaluated, best used
- [ ] Interface: Streamlit UI + FastAPI API
- [ ] Ingestion: fully automated Python pipeline (`python -m ingestion`)
- [ ] Monitoring: user feedback collected + Grafana dashboard with 5+ panels
- [ ] Containerization: docker-compose for everything
- [ ] Reproducibility: instructions clear, dataset auto-downloaded, versions pinned via uv.lock
- [ ] Best practices: hybrid search ✓, document re-ranking ✓, query rewriting ✓

## 16. Implementation order

Work in this order, committing after each phase, keeping tests green:

1. Scaffolding: pyproject, config, docker-compose (qdrant+postgres+grafana), Makefile.
2. Ingestion: download → parse → chunk (+ tests on chunker with a fixture PDF) → index.
3. Retrieval modes + minimal `rag.py` proven in a notebook.
4. Ground truth + retrieval evaluation; pick default config.
5. API + UI + monitoring logging.
6. RAG evaluation + Grafana dashboard.
7. README, screenshots, polish.

## 17. Coding conventions

- Type hints on all public functions; pydantic models at boundaries (API, config).
- No hardcoded secrets, paths, or model names — everything via `config.py`.
- Fail loudly: no bare `except`, no silent skips in the pipeline.
- Every LLM call goes through `providers.py` and is logged with token counts and cost.
- Keep functions small; ingestion stages must be independently runnable and idempotent.