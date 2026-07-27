#!/usr/bin/env bash
# Build + start the full stack on a VM, then ingest the corpus.
# Run from the repo root after editing .env:  bash deploy/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
	cp .env.example .env
	echo "[deploy] created .env from .env.example — edit it and re-run:" >&2
	echo "         set OPENAI_API_KEY, SITE_ADDRESS (your domain, for HTTPS)," >&2
	echo "         and BIND_HOST=127.0.0.1 (keep services off the public net)." >&2
	exit 1
fi

# Safe production defaults if the operator forgot them.
grep -q '^BIND_HOST=' .env || echo 'BIND_HOST=127.0.0.1' >>.env

COMPOSE=(docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml)

echo "[deploy] building images (CPU torch; first build is a few GB)…"
"${COMPOSE[@]}" build

echo "[deploy] starting services…"
"${COMPOSE[@]}" up -d

echo "[deploy] waiting for the API to become healthy…"
until "${COMPOSE[@]}" exec -T api curl -sf http://localhost:8000/health >/dev/null 2>&1; do
	sleep 3
done

echo "[deploy] ingesting the corpus (download → parse → chunk → index)…"
"${COMPOSE[@]}" exec -T api python -m ingestion.flow

echo "[deploy] done. UI:      http(s)://<SITE_ADDRESS>/"
echo "[deploy]       API:     http(s)://<SITE_ADDRESS>/api/health"
echo "[deploy]       Grafana: http(s)://<SITE_ADDRESS>/grafana/"
