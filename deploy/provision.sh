#!/usr/bin/env bash
# One-time provisioning for a fresh Ubuntu/Debian VM: Docker + firewall.
# Run as root (or with sudo):  sudo bash deploy/provision.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
	echo "Run as root: sudo bash deploy/provision.sh" >&2
	exit 1
fi

echo "[provision] installing Docker Engine + compose plugin…"
if ! command -v docker >/dev/null 2>&1; then
	curl -fsSL https://get.docker.com | sh
fi

echo "[provision] configuring firewall (allow SSH, HTTP, HTTPS)…"
if command -v ufw >/dev/null 2>&1; then
	ufw allow OpenSSH || true
	ufw allow 80/tcp
	ufw allow 443/tcp
	ufw --force enable
fi

# Let the invoking user run docker without sudo (re-login required to take effect).
TARGET_USER="${SUDO_USER:-$USER}"
usermod -aG docker "$TARGET_USER" || true

echo "[provision] done. Log out and back in (for docker group), then run deploy/deploy.sh"
