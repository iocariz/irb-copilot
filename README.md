# IRB Copilot

A RAG-based Q&A assistant over public EU prudential regulation for credit-risk
modeling (IRB / definition of default / loan origination). Every answer cites its
sources by document + paragraph. See [SPEC.md](SPEC.md) for the full specification.

> **Status:** under construction. Phase 1 (scaffolding: config, docker-compose,
> Makefile) is in place; ingestion, retrieval, API/UI, evaluation and monitoring
> follow (see SPEC §16). This README is expanded in the final phase (SPEC §14).

## Quick start (dev)

```bash
cp .env.example .env      # then add your OPENAI_API_KEY
make setup                # uv sync + create .env
make up                   # start qdrant + postgres + grafana
make test                 # run the test suite
```

## Layout

See [SPEC.md §3](SPEC.md). Optional torch-backed dependencies (`docling` for PDF
parsing, `sentence-transformers` for reranking) are installed via extras:

```bash
uv sync --extra parse    # ingestion parse stage
uv sync --extra rerank   # RETRIEVAL_MODE=hybrid_rerank
```
