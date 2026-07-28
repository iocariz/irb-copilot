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

# Grafana is exposed via Caddy at /grafana — refuse to deploy with the default
# password (compose already fails if GRAFANA_PASSWORD is unset/empty).
if grep -Eq '^GRAFANA_PASSWORD=(admin)?$' .env; then
	echo "[deploy] refusing to deploy: set a strong GRAFANA_PASSWORD in .env" >&2
	echo "         (it is publicly reachable at /grafana behind Caddy)." >&2
	exit 1
fi

# API auth is enforced in production (REQUIRE_API_KEY=true in the prod overlay);
# the API won't start without a key, so surface it now rather than at boot.
if ! grep -Eq '^API_KEY=.+' .env; then
	echo "[deploy] refusing to deploy: set API_KEY in .env" >&2
	echo "         (production requires it; write endpoints are otherwise open)." >&2
	exit 1
fi

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
