"""Tests for oprim.timeseries.equity_curve_segment_label (A8)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from oprim import equity_curve_segment_label, time_series_split


def _make_equity(start: date, n: int) -> pd.DataFrame:
    dates = [start + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({"date": dates, "equity": [100.0 + i * 0.5 for i in range(n)]})


class TestEquityCurveSegmentLabel:
    def test_standard_three_segments(self) -> None:
        df = _make_equity(date(2024, 1, 1), 100)
        split_dates = {
            "train_end": date(2024, 3, 1),
            "val_start": date(2024, 3, 1),
            "val_end": date(2024, 3, 21),
            "oos_start": date(2024, 3, 21),
        }
        result = equity_curve_segment_label(equity_curve=df, split_dates=split_dates)

        assert set(result.columns) == {"date", "equity", "segment"}
        segments = result["segment"].unique()
        assert "train" in segments
        assert "val" in segments
        assert "oos" in segments

    def test_missing_split_key_raises(self) -> None:
        df = _make_equity(date(2024, 1, 1), 50)
        with pytest.raises(KeyError, match="Missing required"):
            equity_curve_segment_label(
                equity_curve=df,
                split_dates={"train_end": date(2024, 2, 1)},
            )

    def test_equity_curve_short_of_segment_warns(self) -> None:
        # Equity curve that doesn't cover oos period — should still work, just no oos rows
        df = _make_equity(date(2024, 1, 1), 30)
        split_dates = {
            "train_end": date(2024, 1, 20),
            "val_start": date(2024, 1, 20),
            "val_end": date(2024, 1, 25),
            "oos_start": date(2024, 2, 15),  # beyond data
        }
        result = equity_curve_segment_label(equity_curve=df, split_dates=split_dates)
        assert "oos" not in result["segment"].values

    def test_boundary_date_assignment(self) -> None:
        """Right-open: train_end date goes to gap, val_start goes to val."""
        df = _make_equity(date(2024, 1, 1), 100)
        split_dates = {
            "train_end": date(2024, 2, 10),
            "val_start": date(2024, 2, 15),
            "val_end": date(2024, 3, 10),
            "oos_start": date(2024, 3, 10),
        }
        result = equity_curve_segment_label(equity_curve=df, split_dates=split_dates)

        # train_end date should be gap (right-open: train = [start, train_end))
        row_train_end = result[result["date"] == date(2024, 2, 10)]
        assert row_train_end.iloc[0]["segment"] == "gap"

        # val_start date should be val
        row_val_start = result[result["date"] == date(2024, 2, 15)]
        assert row_val_start.iloc[0]["segment"] == "val"

    def test_a7_a8_integration(self) -> None:
        """End-to-end: A7 output feeds A8 input."""
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(200)]
        splits = time_series_split(dates=dates, train_pct=0.6, val_pct=0.2, gap_days=5)

        equity = pd.DataFrame({
            "date": dates,
            "equity": [100.0 + i * 0.3 for i in range(200)],
        })

        result = equity_curve_segment_label(
            equity_curve=equity,
            split_dates=splits["split_dates"],
        )

        assert len(result) == 200
        assert set(result.columns) == {"date", "equity", "segment"}
        # Should have gap segment due to gap_days=5
        assert "gap" in result["segment"].values
        # Segments should be contiguous
        segments_order = result["segment"].tolist()
        # First segment should be train
        assert segments_order[0] == "train"

    def test_series_input(self) -> None:
        """Accept pd.Series with date index."""
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        series = pd.Series([100.0 + i for i in range(50)], index=dates)
        split_dates = {
            "train_end": date(2024, 1, 20),
            "val_start": date(2024, 1, 20),
            "val_end": date(2024, 2, 5),
            "oos_start": date(2024, 2, 5),
        }
        result = equity_curve_segment_label(equity_curve=series, split_dates=split_dates)
        assert len(result) == 50
        assert "train" in result["segment"].values

    def test_dataframe_with_index_no_date_column(self) -> None:
        """Accept DataFrame with date index but no 'date' column."""
        dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame({"equity": [100.0 + i for i in range(50)]}, index=dates)
        split_dates = {
            "train_end": date(2024, 1, 20),
            "val_start": date(2024, 1, 20),
            "val_end": date(2024, 2, 5),
            "oos_start": date(2024, 2, 5),
        }
        result = equity_curve_segment_label(equity_curve=df, split_dates=split_dates)
        assert len(result) == 50
