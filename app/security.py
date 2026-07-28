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


def client_ip(forwarded_for: str | None, fallback: str, trusted_hops: int = 1) -> str:
    """Resolve the real client IP behind ``trusted_hops`` trusted reverse proxies.

    A trusted proxy (e.g. Caddy) *appends* the peer address it saw to
    ``X-Forwarded-For``, so the real client is the ``trusted_hops``-th entry from
    the RIGHT — the rightmost value a client cannot forge. The leftmost entries
    are attacker-supplied: trusting them (as taking the first hop did) lets a
    client rotate the rate-limit key at will and bypass the limiter while growing
    the in-memory map. Anything left of the trusted hops is ignored.

    Falls back to the direct peer when there is no header, ``trusted_hops`` is 0
    (no proxy in front), or fewer hops are present than expected (fails closed to
    the proxy's address rather than an attacker-controlled value).
    """
    if forwarded_for and trusted_hops > 0:
        hops = [h.strip() for h in forwarded_for.split(",") if h.strip()]
        if len(hops) >= trusted_hops:
            return hops[-trusted_hops]
    return fallback
