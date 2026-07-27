"""Lightweight API protections: per-client rate limiting + client-IP resolution.

In-memory and per-process — fine for a single-VM deployment. For multiple
workers/instances, back the limiter with Redis instead.
"""

from __future__ import annotations

import threading


class RateLimiter:
    """Sliding-window request limiter keyed by an arbitrary client id."""

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float) -> bool:
        """Record a hit and return whether it is within the limit (`now` injected)."""
        if self.per_minute <= 0:  # disabled
            return True
        cutoff = now - 60.0
        with self._lock:
            hits = [t for t in self._hits.get(key, ()) if t > cutoff]
            allowed = len(hits) < self.per_minute
            if allowed:
                hits.append(now)
            self._hits[key] = hits
            return allowed


def client_ip(forwarded_for: str | None, fallback: str) -> str:
    """Resolve the real client IP: the first hop of ``X-Forwarded-For`` (set by
    the Caddy reverse proxy) if present, otherwise the direct peer address."""
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    return fallback
