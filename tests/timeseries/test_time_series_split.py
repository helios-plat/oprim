"""Tests for oprim.timeseries.time_series_split (A7)."""

from datetime import date, timedelta

import pytest

from oprim import time_series_split


def _make_dates(start: date, n: int) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


class TestTimeSeriesSplit:
    def test_standard_60_20_20(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 100)
        result = time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=0)

        assert result["n_train"] == 60
        assert result["train"][0] == date(2024, 1, 1)
        assert result["train"][1] == date(2024, 2, 29)  # day index 59
        assert result["val"][0] == date(2024, 3, 1)  # day index 60
        assert result["oos"][1] == date(2024, 4, 9)  # day index 99
        assert result["gap_days"] == 0

    def test_custom_ratio_70_15_15(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 200)
        result = time_series_split(dates=dates, train_pct=0.7, val_pct=0.15, gap_days=0)

        assert result["n_train"] == 140
        assert result["n_val"] + result["n_oos"] == 60
        assert result["train"][1] < result["val"][0]
        assert result["val"][1] < result["oos"][0]

    def test_gap_days_15_no_overlap(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 200)
        result = time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=15)

        assert result["gap_days"] == 15
        train_end = result["split_dates"]["train_end"]
        val_start = result["split_dates"]["val_start"]
        gap = (val_start - train_end).days
        assert gap > 15
        # No overlap between segments
        assert result["train"][1] < result["val"][0]
        assert result["val"][1] < result["oos"][0]

    def test_short_dates_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            time_series_split(dates=[date(2024, 1, 1), date(2024, 1, 2)], train_pct=0.6, val_pct=0.2)

    def test_invalid_ratio_raises(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 100)
        with pytest.raises(ValueError, match="<= 1.0"):
            time_series_split(dates=dates, train_pct=0.8, val_pct=0.3)

    def test_gap_zero_continuous(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 50)
        result = time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=0)

        train_end = result["split_dates"]["train_end"]
        val_start = result["split_dates"]["val_start"]
        # With gap=0, val_start should be the next day after train_end
        assert (val_start - train_end).days == 1

    def test_gap_too_large_raises(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 20)
        with pytest.raises(ValueError, match="gap_days too large"):
            time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=100)

    def test_tiny_pct_raises(self) -> None:
        dates = _make_dates(date(2024, 1, 1), 10)
        with pytest.raises(ValueError, match="at least 1 sample"):
            time_series_split(dates=dates, train_pct=0.05, val_pct=0.05)
