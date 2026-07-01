from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from oprim.should_throttle import should_throttle


# helper
def _now() -> datetime:
    return datetime.now(UTC)


def test_none_last_fired_returns_false() -> None:
    assert should_throttle(last_fired_at=None, throttle_seconds=60) is False


def test_within_window_returns_true() -> None:
    last = _now() - timedelta(seconds=30)
    assert should_throttle(last_fired_at=last, throttle_seconds=60) is True


def test_outside_window_returns_false() -> None:
    last = _now() - timedelta(seconds=90)
    assert should_throttle(last_fired_at=last, throttle_seconds=60) is False


def test_exactly_at_boundary_returns_false() -> None:
    # elapsed == throttle_seconds: NOT strictly less → False (allow)
    anchor = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    last = anchor
    now = anchor + timedelta(seconds=60)
    assert should_throttle(last_fired_at=last, throttle_seconds=60, now=now) is False


def test_one_second_before_boundary_returns_true() -> None:
    anchor = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    last = anchor
    now = anchor + timedelta(seconds=59)
    assert should_throttle(last_fired_at=last, throttle_seconds=60, now=now) is True


def test_zero_throttle_raises() -> None:
    with pytest.raises(ValueError, match="throttle_seconds must be > 0"):
        should_throttle(last_fired_at=None, throttle_seconds=0)


def test_negative_throttle_raises() -> None:
    with pytest.raises(ValueError):
        should_throttle(last_fired_at=None, throttle_seconds=-1)


def test_naive_last_fired_raises() -> None:
    naive = datetime(2026, 1, 1, 0, 0, 0)  # no tzinfo
    with pytest.raises(ValueError, match="timezone-aware"):
        should_throttle(last_fired_at=naive, throttle_seconds=60)


def test_naive_now_raises() -> None:
    last = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    naive_now = datetime(2026, 1, 1, 0, 1, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        should_throttle(last_fired_at=last, throttle_seconds=60, now=naive_now)
