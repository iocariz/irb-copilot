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


def test_client_ip_prefers_forwarded_for_first_hop() -> None:
    assert client_ip("1.2.3.4, 5.6.7.8", "10.0.0.1") == "1.2.3.4"
    assert client_ip(None, "10.0.0.1") == "10.0.0.1"
    assert client_ip("   ", "10.0.0.1") == "10.0.0.1"
