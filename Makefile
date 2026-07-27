# IRB Copilot — developer entry points.
# All Python runs through `uv run` so the pinned environment (uv.lock) is used.

.DEFAULT_GOAL := help
.PHONY: help setup up down ingest eval-retrieval eval-rag run api ui test lint fmt

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Install deps (dev + parse + rerank are default groups) and create .env
	uv sync
	@test -f .env || cp .env.example .env

up: ## Start backing services (qdrant + postgres + grafana)
	docker compose up -d

down: ## Stop backing services
	docker compose down

ingest: ## Run the full ingestion pipeline (download -> parse -> chunk -> index)
	uv run python -m ingestion

eval-retrieval: ## Evaluate retrieval configs, pick the best (writes results/)
	uv run python -m evaluation.eval_retrieval

eval-rag: ## Evaluate RAG prompt/model configs (writes results/)
	uv run python -m evaluation.eval_rag

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
