# Multi-stage, uv-based build. Runs the API by default; the same image runs the
# UI and the ingestion pipeline (docling included) via compose command overrides.

FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Install dependencies first for layer caching. Drop dev tools; keep the parse
# and rerank groups so `docker compose exec api python -m ingestion` (docling)
# and RETRIEVAL_MODE=hybrid_rerank work inside the container.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-group dev
COPY . .

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
USER appuser
EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
