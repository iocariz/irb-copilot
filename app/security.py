"""Lightweight API protections: per-client rate limiting + client-IP resolution.

In-memory and per-process — fine for a single-VM deployment. For multiple
workers/instances, back the limiter with Redis instead.
"""

from __future__ import annotations

import threading

# Cap on distinct client keys held in memory. Without a bound, an attacker who
# rotates source IPs adds a new key per request and never revisits it, growing
# the map without limit (memory-exhaustion DoS). Stale keys are swept first; a
# genuine flood past the cap drops the least-recently-active keys.
_DEFAULT_MAX_KEYS = 100_000


class RateLimiter:
    """Sliding-window request limiter keyed by an arbitrary client id.

    Bounded in memory: a key whose window empties is removed, and the total
    number of keys is capped, so a stream of unique keys can't grow the map
    without limit.
    """

    def __init__(self, per_minute: int, max_keys: int = _DEFAULT_MAX_KEYS) -> None:
        self.per_minute = per_minute
        self.max_keys = max_keys
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
            self._hits[key] = hits  # always non-empty here (a new/expired key is allowed)
            if len(self._hits) > self.max_keys:
                self._evict(cutoff)
            return allowed

    def _evict(self, cutoff: float) -> None:
        """Bound the key map so rotating client ids can't grow it without limit:
        drop windows with no live hits first, then, if still over capacity, the
        least-recently-active keys. Caller holds the lock."""
        stale = [k for k, ts in self._hits.items() if not ts or ts[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
        overflow = len(self._hits) - self.max_keys
        if overflow > 0:  # active flood beyond the cap: drop oldest last-activity
            oldest = sorted(self._hits, key=lambda k: self._hits[k][-1])[:overflow]
            for k in oldest:
                del self._hits[k]


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
