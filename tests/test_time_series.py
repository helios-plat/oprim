"""Tests for oprim.time_series module."""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from oprim.time_series import (
    cumulative_returns,
    ewma_smooth,
    gap_detect,
    lag_forward_fill,
    log_returns,
    percentile_rank,
    purge_embargo_split,
    realized_vol,
    resample_align,
    rolling_window_split,
    zscore_normalize,
)


# ============================================================
# log_returns
# ============================================================
class TestLogReturns:
    def test_simple_increasing(self):
        prices = pd.Series([100.0, 102.0, 104.0, 106.0, 108.0])
        result = log_returns(prices, periods=[1])
        expected = np.log(prices) - np.log(prices.shift(1))
        np.testing.assert_allclose(result["log_ret_1d"].values[1:], expected.values[1:], rtol=1e-9)

    def test_simple_decreasing(self):
        prices = pd.Series([110.0, 105.0, 100.0, 95.0, 90.0])
        result = log_returns(prices, periods=[1])
        assert all(result["log_ret_1d"].dropna() < 0)

    def test_multi_period(self):
        prices = pd.Series([100, 102, 99, 105, 110, 108, 112, 115, 120, 118.0])
        result = log_returns(prices, periods=[1, 5])
        assert "log_ret_1d" in result.columns
        assert "log_ret_5d" in result.columns
        # First 5 rows of 5d should be NaN
        assert result["log_ret_5d"].iloc[:5].isna().all()

    def test_academic_validation(self):
        """Compare with numpy.diff(np.log(prices)) for 1d returns."""
        prices = pd.Series([100, 102, 99, 105, 110.0])
        result = log_returns(prices, periods=[1])
        expected = np.diff(np.log(prices.values))
        np.testing.assert_allclose(result["log_ret_1d"].dropna().values, expected, rtol=1e-9)

    def test_single_element(self):
        prices = pd.Series([100.0])
        result = log_returns(prices, periods=[1])
        assert result["log_ret_1d"].isna().all()

    def test_nan_skip(self):
        prices = pd.Series([100.0, np.nan, 104.0, 106.0])
        result = log_returns(prices, periods=[1], handle_gaps="skip")
        assert result["log_ret_1d"].iloc[1] is pd.NaT or np.isnan(result["log_ret_1d"].iloc[1])

    def test_nan_interpolate(self):
        prices = pd.Series([100.0, np.nan, 104.0, 106.0])
        result = log_returns(prices, periods=[1], handle_gaps="interpolate")
        # After interpolation, NaN at index 1 becomes 102
        assert not np.isnan(result["log_ret_1d"].iloc[2])

    def test_nan_raise(self):
        prices = pd.Series([100.0, np.nan, 104.0])
        with pytest.raises(ValueError, match="NaN"):
            log_returns(prices, periods=[1], handle_gaps="raise")

    def test_zero_price_raises(self):
        prices = pd.Series([100.0, 0.0, 104.0])
        with pytest.raises(ValueError, match="positive"):
            log_returns(prices, periods=[1])

    def test_negative_price_raises(self):
        prices = pd.Series([100.0, -5.0, 104.0])
        with pytest.raises(ValueError, match="positive"):
            log_returns(prices, periods=[1])

    def test_empty_periods_raises(self):
        prices = pd.Series([100.0, 102.0])
        with pytest.raises(ValueError, match="empty"):
            log_returns(prices, periods=[])

    def test_negative_period_raises(self):
        prices = pd.Series([100.0, 102.0])
        with pytest.raises(ValueError, match="periods must be >= 1"):
            log_returns(prices, periods=[0])

    def test_short_series_warning(self):
        prices = pd.Series([100.0, 102.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = log_returns(prices, periods=[1, 5, 20])
            assert len(w) == 1
            assert result["log_ret_20d"].isna().all()

    def test_performance(self, benchmark):
        prices = pd.Series(np.exp(np.cumsum(np.random.default_rng(42).normal(0.0001, 0.01, 10000))))
        benchmark(log_returns, prices, [1, 5, 20, 60])


# ============================================================
# cumulative_returns
# ============================================================
class TestCumulativeReturns:
    def test_log_returns_equity(self):
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        result = cumulative_returns(returns, return_type="log")
        expected = np.exp(returns.cumsum())
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-9)

    def test_simple_compound(self):
        returns = pd.Series([0.1, 0.1, 0.1])
        result = cumulative_returns(returns, return_type="simple", compound=True)
        expected = (1 + returns).cumprod()
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-9)

    def test_simple_non_compound(self):
        returns = pd.Series([0.1, 0.1, 0.1])
        result = cumulative_returns(returns, return_type="simple", compound=False)
        expected = 1 + returns.cumsum()
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-9)

    def test_zero_returns(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        result = cumulative_returns(returns, initial_capital=100.0)
        np.testing.assert_allclose(result.values, [100.0, 100.0, 100.0], rtol=1e-9)

    def test_initial_capital(self):
        returns = pd.Series([0.01])
        result = cumulative_returns(returns, return_type="log", initial_capital=1000.0)
        assert result.iloc[0] == pytest.approx(1000 * np.exp(0.01), rel=1e-9)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="initial_capital"):
            cumulative_returns(pd.Series([0.01]), initial_capital=-1)

    def test_extreme_loss_warning(self):
        returns = pd.Series([0.1, -1.5, 0.1])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cumulative_returns(returns, return_type="simple")
            assert len(w) == 1


# ============================================================
# rolling_window_split
# ============================================================
class TestRollingWindowSplit:
    def test_standard(self):
        result = rolling_window_split(100, 20, step=1)
        assert result[0] == (0, 19)
        assert len(result) == 81

    def test_step_5(self):
        result = rolling_window_split(100, 20, step=5)
        assert result[0] == (0, 19)
        assert result[1] == (5, 24)

    def test_non_overlapping(self):
        result = rolling_window_split(100, 20, step=20)
        assert len(result) == 5
        assert result[0] == (0, 19)
        assert result[1] == (20, 39)

    def test_exact_fit(self):
        result = rolling_window_split(20, 20)
        assert result == [(0, 19)]

    def test_too_short_no_partial(self):
        result = rolling_window_split(10, 20, include_partial=False)
        assert result == []

    def test_too_short_with_partial(self):
        result = rolling_window_split(10, 20, include_partial=True)
        assert result == [(0, 9)]

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            rolling_window_split(-1, 20)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            rolling_window_split(100, 0)

    def test_zero_step_raises(self):
        with pytest.raises(ValueError):
            rolling_window_split(100, 20, step=0)

    def test_performance(self, benchmark):
        benchmark(rolling_window_split, 100000, 252, 21)


# ============================================================
# lag_forward_fill
# ============================================================
class TestLagForwardFill:
    def test_simple_fill(self):
        data = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0])
        result = lag_forward_fill(data, max_gap=5)
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 1.0

    def test_fill_with_lag(self):
        data = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
        result = lag_forward_fill(data, max_gap=5, lag=1)
        # Fill first, then shift
        assert np.isnan(result.iloc[0])

    def test_dataframe(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]})
        result = lag_forward_fill(df, max_gap=5)
        assert result["a"].iloc[1] == 1.0

    def test_all_nan(self):
        data = pd.Series([np.nan, np.nan, np.nan])
        result = lag_forward_fill(data, max_gap=5)
        assert result.isna().all()

    def test_strict_raises(self):
        data = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 8.0])
        with pytest.raises(ValueError, match="Gap"):
            lag_forward_fill(data, max_gap=5, strict=True)

    def test_strict_no_raise(self):
        data = pd.Series([1.0, np.nan, np.nan, 4.0])
        result = lag_forward_fill(data, max_gap=5, strict=True)
        assert result.iloc[1] == 1.0


# ============================================================
# percentile_rank
# ============================================================
class TestPercentileRank:
    def test_expanding(self):
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = percentile_rank(data, method="expanding")
        # Last element should be 1.0 (highest rank)
        assert result.iloc[-1] == 1.0

    def test_rolling(self):
        data = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0, 6.0])
        result = percentile_rank(data, window=3, method="rolling")
        # At index 5, window is [2, 1, 6], rank of 6 is 3/3 = 1.0
        assert result.iloc[5] == 1.0

    def test_cross_sectional(self):
        df = pd.DataFrame({"a": [1.0, 4.0], "b": [2.0, 3.0], "c": [3.0, 2.0]})
        result = percentile_rank(df, method="cross_sectional")
        # Row 0: a=1(rank1), b=2(rank2), c=3(rank3) → 1/3, 2/3, 3/3
        np.testing.assert_allclose(result.iloc[0].values, [1 / 3, 2 / 3, 1.0], rtol=1e-9)

    def test_cross_sectional_academic(self):
        """Compare with scipy.stats.rankdata."""
        df = pd.DataFrame({"a": [10.0], "b": [20.0], "c": [15.0]})
        result = percentile_rank(df, method="cross_sectional")
        expected = rankdata([10, 20, 15], method="average") / 3
        np.testing.assert_allclose(result.iloc[0].values, expected, rtol=1e-9)

    def test_rolling_requires_window(self):
        with pytest.raises(ValueError, match="window"):
            percentile_rank(pd.Series([1, 2, 3]), method="rolling")

    def test_cross_sectional_requires_df(self):
        with pytest.raises(ValueError, match="DataFrame"):
            percentile_rank(pd.Series([1, 2, 3]), method="cross_sectional")


# ============================================================
# ewma_smooth
# ============================================================
class TestEwmaSmooth:
    def test_half_life(self):
        data = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        result = ewma_smooth(data, half_life=10.0)
        expected = data.ewm(halflife=10.0, adjust=True).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_span(self):
        data = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        result = ewma_smooth(data, span=20)
        expected = data.ewm(span=20, adjust=True).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_alpha(self):
        data = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        result = ewma_smooth(data, alpha=0.1)
        expected = data.ewm(alpha=0.1, adjust=True).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_academic_validation(self):
        """Compare with pandas ewm."""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ewma_smooth(data, span=3)
        expected = data.ewm(span=3).mean()
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-9)

    def test_multiple_params_raises(self):
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Exactly one"):
            ewma_smooth(data, half_life=10, span=20)

    def test_no_params_raises(self):
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Exactly one"):
            ewma_smooth(data)

    def test_zero_half_life_raises(self):
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="half_life"):
            ewma_smooth(data, half_life=0)

    def test_performance(self, benchmark):
        data = pd.Series(np.random.default_rng(42).normal(0, 1, 10000))
        benchmark(ewma_smooth, data, half_life=10.0)


# ============================================================
# realized_vol
# ============================================================
class TestRealizedVol:
    def test_close_to_close(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        result = realized_vol(returns, window=20)
        expected = returns.rolling(20).std() * np.sqrt(252)
        np.testing.assert_allclose(result.dropna().values, expected.dropna().values, rtol=1e-9)

    def test_garman_klass(self):
        rng = np.random.default_rng(42)
        n = 100
        ohlc = pd.DataFrame({
            "open": 100 + rng.normal(0, 1, n),
            "high": 102 + rng.normal(0, 1, n),
            "low": 98 + rng.normal(0, 1, n),
            "close": 100 + rng.normal(0, 1, n),
        })
        ohlc["high"] = ohlc[["open", "high", "close"]].max(axis=1) + 0.5
        ohlc["low"] = ohlc[["open", "low", "close"]].min(axis=1) - 0.5
        returns = pd.Series(rng.normal(0, 0.02, n))
        result = realized_vol(returns, window=20, estimator="garman_klass", ohlc=ohlc)
        assert result.dropna().shape[0] > 0
        assert (result.dropna() > 0).all()

    def test_parkinson(self):
        rng = np.random.default_rng(42)
        n = 100
        ohlc = pd.DataFrame({
            "open": 100 + rng.normal(0, 1, n),
            "high": 103 + np.abs(rng.normal(0, 1, n)),
            "low": 97 - np.abs(rng.normal(0, 1, n)),
            "close": 100 + rng.normal(0, 1, n),
        })
        returns = pd.Series(rng.normal(0, 0.02, n))
        result = realized_vol(returns, window=20, estimator="parkinson", ohlc=ohlc)
        assert (result.dropna() > 0).all()

    def test_yang_zhang(self):
        rng = np.random.default_rng(42)
        n = 100
        ohlc = pd.DataFrame({
            "open": 100 + rng.normal(0, 1, n),
            "high": 103 + np.abs(rng.normal(0, 1, n)),
            "low": 97 - np.abs(rng.normal(0, 1, n)),
            "close": 100 + rng.normal(0, 1, n),
        })
        returns = pd.Series(rng.normal(0, 0.02, n))
        result = realized_vol(returns, window=20, estimator="yang_zhang", ohlc=ohlc)
        assert result.dropna().shape[0] > 0

    def test_window_too_small_raises(self):
        with pytest.raises(ValueError, match="window"):
            realized_vol(pd.Series([0.01, 0.02]), window=1)

    def test_no_ohlc_raises(self):
        with pytest.raises(ValueError, match="ohlc"):
            realized_vol(pd.Series([0.01, 0.02]), estimator="garman_klass")

    def test_annualization_factor_zero_raises(self):
        with pytest.raises(ValueError, match="annualization_factor"):
            realized_vol(pd.Series([0.01, 0.02]), annualization_factor=0)

    def test_academic_close_to_close(self):
        """Verify close-to-close matches numpy std * sqrt(252)."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 252))
        window = 20
        result = realized_vol(returns, window=window)
        for i in range(window - 1, len(returns)):
            w = returns.iloc[i - window + 1 : i + 1].values
            expected = np.std(w, ddof=1) * np.sqrt(252)
            np.testing.assert_allclose(result.iloc[i], expected, rtol=1e-9)


# ============================================================
# zscore_normalize
# ============================================================
class TestZscoreNormalize:
    def test_rolling(self):
        data = pd.Series(np.arange(100.0))
        result = zscore_normalize(data, window=20)
        assert result.iloc[:19].isna().all()
        assert not result.iloc[19:].isna().any()

    def test_expanding(self):
        data = pd.Series(np.arange(100.0))
        result = zscore_normalize(data, window=None, min_periods=2)
        assert result.iloc[0].item() is np.nan or np.isnan(result.iloc[0])

    def test_constant_series(self):
        """σ=0 should produce NaN."""
        data = pd.Series([5.0] * 50)
        result = zscore_normalize(data, window=20)
        assert result.iloc[19:].isna().all()

    def test_clip(self):
        data = pd.Series([0.0] * 99 + [1000.0])
        result = zscore_normalize(data, window=50, clip_extreme=5.0)
        assert result.max() <= 5.0

    def test_no_clip(self):
        data = pd.Series([0.0] * 99 + [1000.0])
        result = zscore_normalize(data, window=50, clip_extreme=None)
        assert result.max() > 5.0

    def test_dataframe(self):
        df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0) * 2})
        result = zscore_normalize(df, window=10, min_periods=2)
        assert result.shape == df.shape

    def test_academic_expanding(self):
        """Compare expanding zscore with scipy.stats.zscore (full sample)."""
        from scipy.stats import zscore as scipy_zscore

        data = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
        # Full-sample zscore (expanding at last point ≈ scipy zscore)
        result = zscore_normalize(data, window=None, min_periods=2, clip_extreme=None)
        # At the last point, expanding uses all data
        full_z = scipy_zscore(data.values, ddof=1)
        # They won't be exactly equal (expanding vs full), but last value should be close
        np.testing.assert_allclose(result.iloc[-1], full_z[-1], rtol=1e-6)

    def test_window_lt_min_periods_warning(self):
        data = pd.Series(np.arange(50.0))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            zscore_normalize(data, window=10, min_periods=20)
            assert len(w) == 1


# ============================================================
# gap_detect
# ============================================================
class TestGapDetect:
    def test_no_gaps(self):
        times = pd.date_range("2024-01-01", periods=10, freq="D")
        result = gap_detect(times)
        assert len(result) == 0

    def test_weekend_gap(self):
        # Mon-Fri, then skip weekend, Mon
        times = pd.DatetimeIndex([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08",  # skip weekend
        ])
        result = gap_detect(times, expected_interval=pd.Timedelta(days=1))
        assert len(result) == 1
        assert result.iloc[0]["severity"] in ("short", "medium")

    def test_long_gap(self):
        times = pd.DatetimeIndex(["2024-01-01", "2024-01-15"])
        result = gap_detect(times, expected_interval=pd.Timedelta(days=1))
        assert len(result) == 1
        assert result.iloc[0]["severity"] == "long"

    def test_crypto_thresholds(self):
        times = pd.DatetimeIndex([
            "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 10:00"
        ])
        result = gap_detect(times, expected_interval=pd.Timedelta(hours=1), asset_class="crypto")
        assert len(result) == 1
        assert result.iloc[0]["severity"] == "medium"

    def test_single_timestamp(self):
        times = pd.DatetimeIndex(["2024-01-01"])
        result = gap_detect(times)
        assert len(result) == 0

    def test_identical_timestamps_raises(self):
        times = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-01"])
        with pytest.raises(ValueError, match="identical"):
            gap_detect(times)

    def test_auto_interval(self):
        times = pd.date_range("2024-01-01", periods=50, freq="h")
        # Insert a gap
        times = times.delete(range(10, 20))
        result = gap_detect(times)
        assert len(result) == 1


# ============================================================
# resample_align
# ============================================================
class TestResampleAlign:
    def test_single_frame(self):
        df = pd.DataFrame(
            {"price": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        )
        result = resample_align({"btc": df}, target_freq="D")
        assert "btc_price" in result.columns

    def test_multi_frame(self):
        df1 = pd.DataFrame(
            {"price": [100, 101, 102, 103, 104]},
            index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        )
        df2 = pd.DataFrame(
            {"price": [50, 51, 52, 53, 54]},
            index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        )
        result = resample_align({"stock": df1, "crypto": df2}, target_freq="D")
        assert "stock_price" in result.columns
        assert "crypto_price" in result.columns

    def test_different_freq(self):
        df_daily = pd.DataFrame(
            {"price": range(5)},
            index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
        )
        df_hourly = pd.DataFrame(
            {"price": range(120)},
            index=pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC"),
        )
        result = resample_align({"daily": df_daily, "hourly": df_hourly}, target_freq="D")
        assert len(result) >= 5

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError):
            resample_align({})

    def test_forward_fill(self):
        df = pd.DataFrame(
            {"price": [100, np.nan, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        )
        result = resample_align({"x": df}, target_freq="D", forward_fill_limit=5)
        # NaN should be forward-filled
        assert not result.isna().all().all()


# ============================================================
# purge_embargo_split
# ============================================================
class TestPurgeEmbargoSplit:
    def test_basic(self):
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits = purge_embargo_split(times, n_splits=5, embargo_pct=0.01)
        # May have fewer than n_splits if first fold(s) have empty train
        assert len(splits) >= 4
        for s in splits:
            assert "train" in s
            assert "test" in s
            assert "embargo" in s
            assert len(s["train"]) > 0  # All returned folds must have train data

    def test_no_overlap(self):
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits = purge_embargo_split(times, n_splits=5, embargo_pct=0.01)
        for s in splits:
            train_set = set(s["train"])
            test_set = set(s["test"])
            embargo_set = set(s["embargo"])
            assert train_set.isdisjoint(test_set)
            assert train_set.isdisjoint(embargo_set)

    def test_embargo_zero(self):
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits = purge_embargo_split(times, n_splits=5, embargo_pct=0.0)
        for s in splits:
            assert len(s["embargo"]) == 0

    def test_n_splits_2(self):
        times = pd.date_range("2024-01-01", periods=50, freq="D")  # Larger sample for n_splits=2
        splits = purge_embargo_split(times, n_splits=2)
        assert len(splits) >= 1  # May skip first fold if no train

    def test_unsorted_raises(self):
        times = pd.DatetimeIndex(["2024-01-05", "2024-01-01", "2024-01-03"])
        with pytest.raises(ValueError, match="sorted"):
            purge_embargo_split(times, n_splits=2)

    def test_n_splits_lt_2_raises(self):
        times = pd.date_range("2024-01-01", periods=20, freq="D")
        with pytest.raises(ValueError, match="n_splits"):
            purge_embargo_split(times, n_splits=1)

    def test_embargo_pct_out_of_range_raises(self):
        times = pd.date_range("2024-01-01", periods=20, freq="D")
        with pytest.raises(ValueError, match="embargo_pct"):
            purge_embargo_split(times, n_splits=2, embargo_pct=0.5)

    def test_academic_ground_truth(self):
        """Hand-constructed ground truth: n=100, n_splits=5, embargo=0."""
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits = purge_embargo_split(times, n_splits=5, embargo_pct=0.0)
        # Each fold should have 20 test samples
        for s in splits:
            assert len(s["test"]) == 20
            assert len(s["train"]) > 0
        # Test indices are disjoint (first fold skipped, so we have 4 folds = 80 test samples)
        all_test = np.concatenate([s["test"] for s in splits])
        assert len(np.unique(all_test)) == len(all_test)  # No overlap in test sets


# ============================================================
# Additional coverage tests
# ============================================================
class TestLogReturnsExtra:
    def test_list_input(self):
        """Test non-Series input (list)."""
        result = log_returns([100.0, 102.0, 104.0], periods=[1])
        assert not result.empty

    def test_default_periods(self):
        """Test default periods=[1,5,20,60]."""
        prices = pd.Series(np.exp(np.linspace(0, 1, 100)))
        result = log_returns(prices)
        assert "log_ret_1d" in result.columns
        assert "log_ret_60d" in result.columns


class TestCumulativeReturnsExtra:
    def test_list_input(self):
        """Test non-Series input."""
        result = cumulative_returns([0.01, 0.02, -0.01])
        assert len(result) == 3


class TestLagForwardFillExtra:
    def test_dataframe_strict(self):
        """Test strict mode with DataFrame."""
        df = pd.DataFrame({
            "a": [1.0] + [np.nan] * 10 + [12.0],
            "b": [1.0, 2.0] + [np.nan] * 10,
        })
        with pytest.raises(ValueError, match="Gap"):
            lag_forward_fill(df, max_gap=5, strict=True)

    def test_timedelta_max_gap(self):
        """Test with Timedelta max_gap (uses None limit for ffill)."""
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        data = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0], index=idx)
        result = lag_forward_fill(data, max_gap=pd.Timedelta(days=2))
        # With Timedelta, limit=None so all get filled
        assert result.iloc[1] == 1.0

    def test_list_input(self):
        """Test non-Series/DataFrame input."""
        result = lag_forward_fill([1.0, np.nan, 3.0], max_gap=5)
        assert len(result) == 3


class TestPercentileRankExtra:
    def test_rolling_dataframe(self):
        """Test rolling with DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
        result = percentile_rank(df, window=3, method="rolling")
        assert result.shape == df.shape

    def test_expanding_dataframe(self):
        """Test expanding with DataFrame."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        result = percentile_rank(df, method="expanding")
        assert result.shape == df.shape

    def test_cross_sectional_nan_row(self):
        """Test cross_sectional with all-NaN row."""
        df = pd.DataFrame({"a": [np.nan, 2.0], "b": [np.nan, 3.0]})
        result = percentile_rank(df, method="cross_sectional")
        assert result.iloc[0].isna().all()


class TestEwmaSmoothExtra:
    def test_invalid_alpha(self):
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="alpha"):
            ewma_smooth(data, alpha=0.0)

    def test_invalid_span(self):
        data = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="span"):
            ewma_smooth(data, span=0)


class TestRealizedVolExtra:
    def test_missing_ohlc_column(self):
        ohlc = pd.DataFrame({"open": [100], "high": [101], "low": [99]})  # missing 'close'
        with pytest.raises(ValueError, match="close"):
            realized_vol(pd.Series([0.01]), window=2, estimator="garman_klass", ohlc=ohlc)


class TestResampleAlignExtra:
    def test_method_mean(self):
        df = pd.DataFrame(
            {"price": range(48)},
            index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
        )
        result = resample_align({"x": df}, target_freq="D", method="mean")
        assert "x_price" in result.columns

    def test_method_ohlc(self):
        df = pd.DataFrame(
            {"price": range(48)},
            index=pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
        )
        result = resample_align({"x": df}, target_freq="D", method="ohlc")
        assert len(result.columns) > 0

    def test_no_tz_input(self):
        """Test DataFrame without timezone gets localized."""
        df = pd.DataFrame(
            {"price": [100, 101, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )
        result = resample_align({"x": df}, target_freq="D")
        assert "x_price" in result.columns

    def test_no_datetime_index_raises(self):
        df = pd.DataFrame({"price": [100, 101]}, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            resample_align({"x": df})

    def test_forward_fill_zero(self):
        df = pd.DataFrame(
            {"price": [100, np.nan, 102]},
            index=pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
        )
        result = resample_align({"x": df}, target_freq="D", forward_fill_limit=0)
        # Should not forward fill
        assert result["x_price"].isna().any()


class TestPurgeEmbargoSplitExtra:
    def test_label_horizon_timedelta(self):
        """Test purge with Timedelta label_horizon."""
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits = purge_embargo_split(times, n_splits=5, label_horizon=pd.Timedelta(days=5))
        # Train should have fewer samples due to purging
        for s in splits:
            total = len(s["train"]) + len(s["test"]) + len(s["embargo"])
            assert total <= 100

    def test_label_horizon_int(self):
        """Test purge with int label_horizon > 0."""
        times = pd.date_range("2024-01-01", periods=100, freq="D")
        splits_no_purge = purge_embargo_split(times, n_splits=5, label_horizon=0)
        splits_purge = purge_embargo_split(times, n_splits=5, label_horizon=5)
        # With purge, train should be smaller
        assert len(splits_purge[1]["train"]) <= len(splits_no_purge[1]["train"])
