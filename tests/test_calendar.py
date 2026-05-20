"""Tests for oprim.calendar — Sprint 0 business day primitives."""

from __future__ import annotations

from datetime import date

import pytest

from oprim.calendar import is_business_day, prev_business_day


class TestIsBusinessDay:
    def test_monday_is_business_day(self) -> None:
        assert is_business_day(date(2026, 5, 18)) is True  # Monday

    def test_tuesday_is_business_day(self) -> None:
        assert is_business_day(date(2026, 5, 19)) is True  # Tuesday

    def test_friday_is_business_day(self) -> None:
        assert is_business_day(date(2026, 5, 15)) is True  # Friday

    def test_saturday_not_business_day(self) -> None:
        assert is_business_day(date(2026, 5, 16)) is False  # Saturday

    def test_sunday_not_business_day(self) -> None:
        assert is_business_day(date(2026, 5, 17)) is False  # Sunday

    @pytest.mark.academic_reference
    def test_academic_weekday_check(self) -> None:
        """Verify weekday() < 5 rule (Mon=0, Sun=6).

        Reference: ISO 8601 business day definition.
        """
        # Monday through Friday = business days
        monday = date(2026, 5, 18)
        for i in range(5):
            d = date.fromordinal(monday.toordinal() + i)
            assert is_business_day(d) is True
        # Saturday, Sunday
        assert is_business_day(date(2026, 5, 23)) is False
        assert is_business_day(date(2026, 5, 24)) is False


class TestPrevBusinessDay:
    def test_tuesday_prev_1_is_monday(self) -> None:
        result = prev_business_day(date(2026, 5, 19), n=1)  # Tuesday → Monday
        assert result == date(2026, 5, 18)

    def test_tuesday_prev_3(self) -> None:
        result = prev_business_day(date(2026, 5, 19), n=3)  # Tuesday → last Thursday
        assert result == date(2026, 5, 14)

    def test_monday_prev_1_is_friday(self) -> None:
        result = prev_business_day(date(2026, 5, 18), n=1)  # Monday → Friday
        assert result == date(2026, 5, 15)

    def test_monday_prev_2_is_thursday(self) -> None:
        result = prev_business_day(date(2026, 5, 18), n=2)  # Monday → Thursday
        assert result == date(2026, 5, 14)

    def test_default_n_is_1(self) -> None:
        result = prev_business_day(date(2026, 5, 19))
        assert result == date(2026, 5, 18)

    def test_raises_on_n_zero(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            prev_business_day(date(2026, 5, 19), n=0)

    def test_raises_on_negative_n(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 1"):
            prev_business_day(date(2026, 5, 19), n=-5)

    def test_from_weekend(self) -> None:
        # Saturday (2026-05-16), prev 1 business day = Friday (2026-05-15)
        result = prev_business_day(date(2026, 5, 16), n=1)
        assert result == date(2026, 5, 15)

    @pytest.mark.academic_reference
    def test_academic_spanning_weekend(self) -> None:
        """Verify correct skipping of weekends.

        Reference: pandas.tseries.offsets.BusinessDay (comparison only).
        """
        # Tuesday May 19 → 5 business days back
        # Tue May 19 → Mon May 18 → Fri May 15 → Thu May 14 → Wed May 13 → Tue May 12
        result = prev_business_day(date(2026, 5, 19), n=5)
        assert result == date(2026, 5, 12)
