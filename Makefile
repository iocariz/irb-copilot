# IRB Copilot — developer entry points.
# All Python runs through `uv run` so the pinned environment (uv.lock) is used.

.DEFAULT_GOAL := help
.PHONY: help setup up up-all down prod-up prod-down ingest ground-truth ground-truth-hard eval-retrieval eval-retrieval-hard eval-rag regen run api ui test lint fmt

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (dev + parse + rerank are default groups) and create .env
	uv sync
	@test -f .env || cp .env.example .env

up: ## Start backing services only (qdrant + postgres + grafana) — pair with `make run`
	docker compose up -d qdrant postgres grafana

up-all: ## Start everything in Docker (backing services + api + ui)
	docker compose up -d

down: ## Stop all services
	docker compose down

prod-up: ## Start the full production stack (Caddy + all services) — see deploy/README.md
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml up -d --build

prod-down: ## Stop the production stack
	docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml down

ingest: ## Run the full ingestion pipeline via Prefect (download -> parse -> chunk -> index)
	uv run python -m ingestion.flow

ground-truth: ## Generate evaluation ground truth (LLM; writes evaluation/ground_truth.csv)
	uv run python -m evaluation.generate_ground_truth

ground-truth-hard: ## Generate de-biased (paraphrased) ground truth
	uv run python -m evaluation.generate_ground_truth --style hard

eval-retrieval: ## Evaluate retrieval configs, pick the best (writes results/)
	uv run python -m evaluation.eval_retrieval

eval-retrieval-hard: ## Evaluate retrieval on the de-biased ground truth
	uv run python -m evaluation.eval_retrieval --ground-truth evaluation/ground_truth_hard.csv

eval-rag: ## Evaluate RAG prompt/model configs (writes results/)
	uv run python -m evaluation.eval_rag

regen: up ## Regenerate EVERYTHING: re-ingest + ground truth + retrieval & RAG eval (Docker + OPENAI_API_KEY)
	@echo "[regen] full pipeline: re-ingest -> ground truth -> retrieval & RAG eval."
	@echo "[regen] requires OPENAI_API_KEY in .env; this makes paid LLM/embedding calls and can take a while."
	@grep -Eq '^OPENAI_API_KEY=.+' .env 2>/dev/null || { echo "[regen] ERROR: set OPENAI_API_KEY in .env first"; exit 1; }
	uv run python -m ingestion.flow --recreate                                        # clean rebuild: chunk ids change, so drop the collection to avoid orphans
	uv run python -m evaluation.generate_ground_truth                                 # writes ground_truth.csv + .meta.json
	uv run python -m evaluation.generate_ground_truth --style hard                    # writes ground_truth_hard.csv + .meta.json
	uv run python -m evaluation.eval_retrieval --rebuild-naive                        # rebuilds the stale naive index once
	uv run python -m evaluation.eval_retrieval --ground-truth evaluation/ground_truth_hard.csv
	uv run python -m evaluation.eval_rag                                              # also re-seeds the Grafana judge panel
	@echo "[regen] done. Review evaluation/results/*.csv; if BEST changed, update RETRIEVAL_MODE/CHUNKER in .env.example + README."

run: ## Run API + UI locally
	$(MAKE) -j2 api ui

api: ## Run the FastAPI backend
	uv run uvicorn app.api:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

ui: ## Run the Streamlit front end
	uv run streamlit run app/ui.py --server.port $${UI_PORT:-8501}

test: ## Run the test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

fmt: ## Format with ruff
	uv run ruff format .
