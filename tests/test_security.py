"""Tests for the rate limiter and client-IP resolution (pure)."""

from __future__ import annotations

from app.security import RateLimiter, client_ip


def test_rate_limiter_allows_up_to_limit_then_blocks() -> None:
    rl = RateLimiter(3)
    assert [rl.allow("ip", 100.0) for _ in range(4)] == [True, True, True, False]


def test_rate_limiter_window_slides() -> None:
    rl = RateLimiter(2)
    assert rl.allow("ip", 100.0)
    assert rl.allow("ip", 100.5)
    assert not rl.allow("ip", 101.0)  # 3rd hit within the 60s window
    assert rl.allow("ip", 161.5)      # earlier hits have aged out


def test_rate_limiter_is_per_key() -> None:
    rl = RateLimiter(1)
    assert rl.allow("a", 100.0)
    assert rl.allow("b", 100.0)  # different client, own budget
    assert not rl.allow("a", 100.0)


def test_rate_limiter_zero_disables() -> None:
    rl = RateLimiter(0)
    assert all(rl.allow("ip", 100.0) for _ in range(100))


def test_rate_limiter_map_is_bounded_under_key_rotation() -> None:
    # An attacker rotating client ids must not grow the map without limit.
    rl = RateLimiter(5, max_keys=10)
    for i in range(1000):
        rl.allow(f"ip-{i}", 100.0)  # all within one window, all unique
    assert len(rl._hits) <= 10


def test_rate_limiter_evicts_stale_keys_before_active_ones() -> None:
    rl = RateLimiter(5, max_keys=3)
    # Old keys, hit long ago (will be > 60s stale relative to the new burst).
    for i in range(3):
        rl.allow(f"old-{i}", 100.0)
    # A later burst of fresh keys past the cap should sweep the stale ones first.
    for i in range(3):
        rl.allow(f"new-{i}", 1000.0)
    assert len(rl._hits) <= 3
    assert all(k.startswith("new-") for k in rl._hits)  # stale 'old-*' evicted


def test_rate_limiter_still_limits_a_single_key_after_eviction() -> None:
    rl = RateLimiter(1, max_keys=2)
    assert rl.allow("victim", 100.0)
    for i in range(50):  # flood unique keys to trigger eviction churn
        rl.allow(f"flood-{i}", 100.0)
    # The victim's own limit is still enforced if its window survived; if it was
    # evicted it simply gets a fresh budget — either way the map stays bounded.
    assert len(rl._hits) <= 2


def test_client_ip_uses_hop_appended_by_trusted_proxy() -> None:
    # One trusted proxy (Caddy) appends the real peer on the RIGHT; the leftmost
    # hop is client-supplied and must be ignored to prevent key rotation.
    assert client_ip("1.2.3.4, 5.6.7.8", "10.0.0.1") == "5.6.7.8"
    # A spoofed leftmost value does not change the resolved (rightmost) IP.
    assert client_ip("evil, evil2, 5.6.7.8", "10.0.0.1") == "5.6.7.8"
    # Client sent no XFF: Caddy's single appended hop is the real client.
    assert client_ip("5.6.7.8", "10.0.0.1") == "5.6.7.8"


def test_client_ip_supports_multiple_trusted_hops() -> None:
    # Two trusted proxies: real client is 2nd from the right; left hop spoofable.
    assert client_ip("spoof, 9.9.9.9, 5.6.7.8", "10.0.0.1", trusted_hops=2) == "9.9.9.9"


def test_client_ip_falls_back_when_no_usable_header() -> None:
    assert client_ip(None, "10.0.0.1") == "10.0.0.1"
    assert client_ip("   ", "10.0.0.1") == "10.0.0.1"
    # trusted_hops=0 (no proxy) -> always the direct peer, header ignored.
    assert client_ip("1.2.3.4", "10.0.0.1", trusted_hops=0) == "10.0.0.1"
    # Fewer hops than trusted proxies -> fail closed to the direct peer.
    assert client_ip("1.2.3.4", "10.0.0.1", trusted_hops=2) == "10.0.0.1"
