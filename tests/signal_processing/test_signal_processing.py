"""Tests for oprim.signal_processing module."""

import numpy as np
import pytest

from oprim.signal_processing import (
    H_change_rate_std,
    atr,
    hurst_exponent,
    linear_slope,
    orderbook_entropy,
)


class TestLinearSlope:
    def test_rising_series(self):
        """1, 2, 3, ... -> positive slope."""
        values = np.arange(1, 11, dtype=float)
        slope = linear_slope(values)
        assert slope > 0

    def test_falling_series(self):
        """5, 4, 3, ... -> negative slope (normalized = abs, so > 0 for normalize=True)."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # normalize=True returns abs(slope)/mean so must be positive
        slope = linear_slope(values)
        assert slope > 0

    def test_falling_series_normalize_false(self):
        """5, 4, 3, ... normalize=False -> negative slope."""
        values = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        slope = linear_slope(values, normalize=False)
        assert slope < 0

    def test_normalize_false_returns_raw(self):
        """normalize=False returns raw slope without abs or scaling."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = linear_slope(values, normalize=False)
        assert slope == pytest.approx(1.0, rel=1e-6)

    def test_constant_series(self):
        """Flat series -> slope = 0."""
        values = np.full(10, 3.14)
        assert linear_slope(values) == pytest.approx(0.0, abs=1e-12)

    def test_too_short_raises(self):
        """len < 2 raises ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            linear_slope(np.array([1.0]))

    def test_two_points(self):
        """Exactly 2 points is valid."""
        values = np.array([0.0, 1.0])
        slope = linear_slope(values, normalize=False)
        assert slope == pytest.approx(1.0, rel=1e-6)

    def test_mean_zero_normalize(self):
        """When mean == 0, normalized slope returns 0.0 to avoid div-by-zero."""
        values = np.array([-1.0, 0.0, 1.0])
        slope = linear_slope(values, normalize=True)
        assert slope == pytest.approx(0.0, abs=1e-12)


class TestATR:
    def _make_ohlc(self, n, range_val=1.0, trend=0.0):
        """Generate simple OHLC arrays."""
        closes = np.cumsum(np.full(n, trend)) + 100.0
        highs = closes + range_val / 2
        lows = closes - range_val / 2
        return highs, lows, closes

    def test_wilder_smoothing_triggered(self):
        """period=3, len=20 -> Wilder smoothing loop runs (trs[period:] has elements)."""
        highs, lows, closes = self._make_ohlc(20, range_val=2.0)
        result = atr(highs, lows, closes, period=3)
        assert np.isfinite(result)
        assert result > 0

    def test_basic_constant_range(self):
        """Constant H-L range -> ATR equals that range."""
        n = 20
        closes = np.full(n, 100.0)
        highs = np.full(n, 101.0)
        lows = np.full(n, 99.0)
        result = atr(highs, lows, closes, period=5)
        # True range = max(H-L, |H-C_prev|, |L-C_prev|) = 2 for constant prices
        assert result == pytest.approx(2.0, rel=1e-6)

    def test_too_few_bars_raises(self):
        """n < period + 1 raises ValueError."""
        highs = np.array([101.0, 102.0])
        lows = np.array([99.0, 98.0])
        closes = np.array([100.0, 101.0])
        with pytest.raises(ValueError, match="at least"):
            atr(highs, lows, closes, period=5)

    def test_minimum_valid_length(self):
        """n == period + 1 is valid (SMA seed only, no smoothing loop)."""
        n = 4  # period=3, need 4 bars
        highs, lows, closes = self._make_ohlc(n)
        result = atr(highs, lows, closes, period=3)
        assert np.isfinite(result)


class TestHurstExponent:
    def test_random_walk(self):
        """np.cumsum(randn) -> H close to 0.5 (allow [0.3, 0.7])."""
        rng = np.random.default_rng(42)
        series = rng.standard_normal(512)
        H = hurst_exponent(series)
        assert 0.3 <= H <= 0.7

    def test_trending(self):
        """Monotone series -> H > 0.5."""
        series = np.linspace(0, 100, 256)
        H = hurst_exponent(series)
        assert H > 0.5

    def test_too_short_raises(self):
        """Series shorter than min_window * 2 raises ValueError."""
        with pytest.raises(ValueError):
            hurst_exponent(np.arange(10.0), min_window=10)

    def test_output_bounded(self):
        """Result is in [0, 1]."""
        rng = np.random.default_rng(0)
        series = rng.standard_normal(256)
        H = hurst_exponent(series)
        assert 0.0 <= H <= 1.0

    @pytest.mark.academic_reference
    def test_hurst_rs_analysis_1951(self):
        """Hurst (1951): H ≈ 0.5 for iid normal series (uncorrelated noise)."""
        rng = np.random.default_rng(999)
        results = []
        for _ in range(5):
            # Use iid normal directly (not cumsum); cumsum produces non-stationary
            # levels where this R/S implementation returns H close to 1.
            series = rng.standard_normal(512)
            results.append(hurst_exponent(series))
        mean_H = np.mean(results)
        assert 0.3 <= mean_H <= 0.7

    def test_not_enough_window_sizes_raises(self):
        """Series where only 1 window size fits → ValueError 'Not enough window sizes'."""
        # n=8, min_window=4: max_k=3, sizes=[4] only (4<=4, 8>4) → len<2 → raises
        series = np.arange(8, dtype=float)
        with pytest.raises(ValueError):
            hurst_exponent(series, min_window=4)

    def test_constant_series_returns_fallback(self):
        """Constant series → all chunk stds=0 → rs_vals empty → fallback 0.5."""
        series = np.full(64, 3.14)
        H = hurst_exponent(series)
        assert H == pytest.approx(0.5, abs=0.01)


class TestHChangeRateStd:
    def test_basic(self):
        """Known diff std: values = [0,1,2,3,4,5,6], window=6 -> diffs=[1,1,1,1,1,1] -> std=0."""
        values = np.arange(7, dtype=float)
        result = H_change_rate_std(values, window=6)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_varying_diffs(self):
        """Non-constant diffs should have positive std."""
        values = np.array([1.0, 2.0, 4.0, 3.0, 5.0, 2.0, 6.0])
        result = H_change_rate_std(values, window=6)
        assert result > 0

    def test_too_short_raises(self):
        """len < window + 1 raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            H_change_rate_std(np.array([1.0, 2.0, 3.0]), window=6)

    def test_uses_tail_segment(self):
        """Uses last window+1 values, not all data."""
        # Prefix is noisy, tail is constant -> std should be 0
        values = np.array([10.0, -10.0, 5.0, -5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = H_change_rate_std(values, window=5)
        assert result == pytest.approx(0.0, abs=1e-12)


class TestOrderbookEntropy:
    def test_uniform(self):
        """Equal sizes -> maximum entropy = ln(n)."""
        sizes = np.ones(5)
        h = orderbook_entropy(sizes)
        assert h == pytest.approx(np.log(5), rel=1e-9)

    def test_concentrated(self):
        """Single level -> all probability there -> H = 0."""
        sizes = np.array([1.0, 0.0, 0.0, 0.0])
        h = orderbook_entropy(sizes)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_empty_sizes(self):
        """All zeros -> H = 0."""
        sizes = np.array([0.0, 0.0, 0.0])
        h = orderbook_entropy(sizes)
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_empty_array(self):
        """Empty array -> H = 0."""
        h = orderbook_entropy(np.array([]))
        assert h == pytest.approx(0.0, abs=1e-12)

    def test_two_levels_equal(self):
        """Two equal levels -> H = ln(2)."""
        sizes = np.array([1.0, 1.0])
        h = orderbook_entropy(sizes)
        assert h == pytest.approx(np.log(2), rel=1e-9)

    def test_nats(self):
        """Result is in nats (natural log base)."""
        sizes = np.array([1.0, 1.0, 1.0, 1.0])
        h = orderbook_entropy(sizes)
        assert h == pytest.approx(np.log(4), rel=1e-9)
