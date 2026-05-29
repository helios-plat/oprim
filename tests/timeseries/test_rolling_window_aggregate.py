"""Tests for oprim.timeseries.rolling_window_aggregate (A4)."""

import numpy as np
import pandas as pd
import pytest

from oprim import rolling_window_aggregate


class TestRollingWindowAggregate:
    def test_int_window_all_aggs(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        for agg in ["mean", "std", "min", "max", "sum", "median"]:
            result = rolling_window_aggregate(series=s, window=3, agg=agg)
            assert len(result) == 5

    def test_time_window_7D(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30)
        s = pd.Series(range(30), index=idx)
        result = rolling_window_aggregate(series=s, window="7D", agg="mean")
        assert len(result) == 30

    def test_min_periods_boundary(self) -> None:
        s = pd.Series([1, 2, 3, 4, 5])
        result = rolling_window_aggregate(series=s, window=3, agg="mean", min_periods=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert not pd.isna(result.iloc[2])

    def test_quantile_0_05_50_95(self) -> None:
        s = pd.Series(range(100))
        for q in [0.05, 0.5, 0.95]:
            result = rolling_window_aggregate(series=s, window=20, agg="quantile", quantile_q=q)
            assert len(result) == 100

    def test_numpy_input_compatible(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_window_aggregate(series=arr, window=3, agg="mean")
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_nan_handling(self) -> None:
        s = pd.Series([1, np.nan, 3, 4, 5])
        result = rolling_window_aggregate(series=s, window=3, agg="mean")
        assert len(result) == 5

    def test_invalid_window_raises(self) -> None:
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown agg"):
            rolling_window_aggregate(series=s, window=2, agg="invalid")
