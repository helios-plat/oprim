"""Tests for oprim.window_active_check (aegis DESIGN §3.2/§5.3/§9)."""

from __future__ import annotations

from datetime import datetime

import pytest

from oprim import window_active_check
from oprim._exceptions import OprimValidationError
from oprim._window_active import WindowStatus


class TestWindowNone:
    def test_inside_one_shot(self):
        start = datetime(2026, 7, 4, 10, 0, 0)
        now = datetime(2026, 7, 4, 10, 30, 0)
        r = window_active_check(now=now, start=start, duration_seconds=3600)
        assert isinstance(r, WindowStatus)
        assert r.active is True

    def test_before_and_after(self):
        start = datetime(2026, 7, 4, 10, 0, 0)
        assert not window_active_check(
            now=datetime(2026, 7, 4, 9, 59, 0), start=start, duration_seconds=3600
        ).active
        assert not window_active_check(
            now=datetime(2026, 7, 4, 11, 0, 0), start=start, duration_seconds=3600
        ).active


class TestWindowDaily:
    def test_same_time_next_day(self):
        start = datetime(2026, 7, 4, 2, 0, 0)  # 02:00 nightly
        now = datetime(2026, 8, 1, 2, 30, 0)  # different day, in window
        assert window_active_check(
            now=now, start=start, duration_seconds=3600, recurrence="daily"
        ).active

    def test_outside_time_of_day(self):
        start = datetime(2026, 7, 4, 2, 0, 0)
        now = datetime(2026, 7, 4, 5, 0, 0)
        assert not window_active_check(
            now=now, start=start, duration_seconds=3600, recurrence="daily"
        ).active

    def test_crosses_midnight(self):
        # 23:00 for 4h → active 23:00–03:00
        start = datetime(2026, 7, 4, 23, 0, 0)
        assert window_active_check(
            now=datetime(2026, 7, 5, 1, 0, 0),
            start=start,
            duration_seconds=4 * 3600,
            recurrence="daily",
        ).active
        assert not window_active_check(
            now=datetime(2026, 7, 5, 4, 0, 0),
            start=start,
            duration_seconds=4 * 3600,
            recurrence="daily",
        ).active


class TestWindowWeekly:
    def test_matching_weekday_and_time(self):
        start = datetime(2026, 7, 4, 1, 0, 0)  # 2026-07-04 is Saturday (weekday 5)
        assert start.weekday() == 5
        now = datetime(2026, 7, 11, 1, 30, 0)  # next Saturday, in window
        assert window_active_check(
            now=now, start=start, duration_seconds=3600, recurrence="weekly"
        ).active

    def test_wrong_weekday(self):
        start = datetime(2026, 7, 4, 1, 0, 0)  # Saturday
        now = datetime(2026, 7, 5, 1, 30, 0)  # Sunday
        assert not window_active_check(
            now=now, start=start, duration_seconds=3600, recurrence="weekly"
        ).active

    def test_explicit_weekdays(self):
        start = datetime(2026, 7, 4, 1, 0, 0)
        now = datetime(2026, 7, 6, 1, 30, 0)  # Monday (weekday 0)
        assert window_active_check(
            now=now, start=start, duration_seconds=3600, recurrence="weekly", weekdays=[0, 6]
        ).active

    def test_weekly_cross_midnight_rejected(self):
        start = datetime(2026, 7, 4, 23, 0, 0)
        with pytest.raises(OprimValidationError):
            window_active_check(
                now=start, start=start, duration_seconds=4 * 3600, recurrence="weekly"
            )


class TestWindowValidation:
    def test_zero_duration(self):
        with pytest.raises(OprimValidationError):
            window_active_check(
                now=datetime(2026, 7, 4), start=datetime(2026, 7, 4), duration_seconds=0
            )

    def test_unknown_recurrence(self):
        with pytest.raises(OprimValidationError):
            window_active_check(
                now=datetime(2026, 7, 4),
                start=datetime(2026, 7, 4),
                duration_seconds=60,
                recurrence="monthly",  # type: ignore[arg-type]
            )
