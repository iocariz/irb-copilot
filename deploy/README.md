# Deploying IRB Copilot to a VM

Runs the whole stack (Qdrant, Postgres, Grafana, API, UI) with docker-compose on
a single Linux VM, behind a **Caddy** reverse proxy that is the only thing exposed
to the internet (ports 80/443, automatic HTTPS with a domain). Qdrant, Postgres
and Grafana are reached only over the internal compose network.

```
Internet ──► Caddy :80/:443 ──►  /          → ui:8501   (Streamlit)
                                 /api/*      → api:8000  (FastAPI)
                                 /grafana/*  → grafana:3000
                          (qdrant, postgres: internal only)
```

## Prerequisites
- A VM (2+ vCPU, **4 GB+ RAM**, ~15 GB disk — the CPU-torch image + models are a
  few GB), Ubuntu/Debian.
- An `OPENAI_API_KEY`.
- Optional but recommended: a domain name pointed (A record) at the VM's IP, for
  automatic HTTPS.

## Steps

```bash
# 1. On the VM: install Docker + firewall (once)
git clone https://github.com/iocariz/irb-copilot.git && cd irb-copilot
sudo bash deploy/provision.sh
#    log out and back in so the docker group applies

# 2. Configure
cp .env.example .env
#    edit .env and set:
#      OPENAI_API_KEY=sk-...
#      BIND_HOST=127.0.0.1          # keep app/db/dashboard off the public net
#      SITE_ADDRESS=example.com     # your domain (or ":80" to serve HTTP on the IP)
#      GRAFANA_PASSWORD=<something>  # change from the default!

# 3. Build, start, and ingest
bash deploy/deploy.sh
```

Then browse to:
- **UI** — `https://example.com/`
- **API** — `https://example.com/api/health`
- **Grafana** — `https://example.com/grafana/` (login: `admin` / your `GRAFANA_PASSWORD`)

## Operations

```bash
C="docker compose -f docker-compose.yml -f deploy/docker-compose.prod.yml"
$C ps                 # status
$C logs -f api ui     # tail logs
$C exec api python -m ingestion.flow --from-stage index --recreate   # re-index
$C pull && $C up -d --build                                          # update
$C down               # stop (add -v to wipe volumes/data)
```

## Notes
- **Change the defaults** in `.env` before exposing publicly: `GRAFANA_PASSWORD`,
  `POSTGRES_PASSWORD`.
- The default `RETRIEVAL_MODE=hybrid_rerank` downloads the cross-encoder model on
  the first query (~1 GB); set `RETRIEVAL_MODE=bm25` for a lighter/faster deploy.
- `SITE_ADDRESS=:80` serves plain HTTP on the VM IP (fine for a quick demo);
  a domain enables automatic HTTPS.
