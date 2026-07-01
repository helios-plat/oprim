"""Tests for oprim.technical.trend: atr_series, adx_series, supertrend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oprim.technical.trend import adx_series, atr_series, supertrend


# ──────────────────── Fixtures ────────────────────

@pytest.fixture
def synthetic_ohlcv():
    """100-bar synthetic OHLCV: sine wave trend for controllable testing."""
    rng = np.random.default_rng(42)
    n = 200
    t = np.arange(n, dtype=float)
    close = 100.0 + 10.0 * np.sin(t * 0.1) + rng.normal(0, 0.5, n)
    high  = close + rng.uniform(0.2, 1.5, n)
    low   = close - rng.uniform(0.2, 1.5, n)
    vol   = rng.uniform(100, 1000, n)
    return {"high": high, "low": low, "close": close, "volume": vol}


@pytest.fixture
def uptrend_ohlcv():
    """200-bar persistent uptrend."""
    n = 200
    base = np.linspace(100, 200, n)
    high  = base + 2.0
    low   = base - 2.0
    close = base + 0.5
    return {"high": high, "low": low, "close": close}


@pytest.fixture
def downtrend_ohlcv():
    """200-bar persistent downtrend."""
    n = 200
    base = np.linspace(200, 100, n)
    high  = base + 2.0
    low   = base - 2.0
    close = base - 0.5
    return {"high": high, "low": low, "close": close}


# ──────────────────── atr_series ────────────────────

class TestAtrSeries:
    def test_output_shape(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = atr_series(h, l, c, period=14)
        assert result.shape == (len(c),)

    def test_warmup_nans(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        period = 14
        result = atr_series(h, l, c, period=period)
        # _wilder_atr sets first valid ATR at index `period` (seed_idx+1=period-1+1=period)
        # so indices 0..period-1 must be NaN
        assert np.all(np.isnan(result[:period]))

    def test_positive_values(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = atr_series(h, l, c, period=14)
        valid = result[~np.isnan(result)]
        assert np.all(valid > 0)

    def test_series_input_preserves_type(self, synthetic_ohlcv):
        idx = pd.date_range("2020-01-01", periods=len(synthetic_ohlcv["close"]), freq="1H")
        h = pd.Series(synthetic_ohlcv["high"], index=idx)
        l = pd.Series(synthetic_ohlcv["low"], index=idx)
        c = pd.Series(synthetic_ohlcv["close"], index=idx)
        result = atr_series(h, l, c, period=14)
        assert isinstance(result, pd.Series)
        assert result.index.equals(idx)

    def test_different_periods(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        atr_fast = atr_series(h, l, c, period=7)
        atr_slow = atr_series(h, l, c, period=28)
        # Fast ATR is more reactive → higher variance in valid region
        fast_valid = atr_fast[~np.isnan(atr_fast)]
        slow_valid = atr_slow[~np.isnan(atr_slow)]
        assert float(np.std(fast_valid)) >= float(np.std(slow_valid)) * 0.8  # fast ≥ slow variance

    def test_invalid_period_raises(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        with pytest.raises(ValueError, match="period"):
            atr_series(h, l, c, period=0)

    def test_mismatched_lengths_raises(self, synthetic_ohlcv):
        h = synthetic_ohlcv["high"]
        l = synthetic_ohlcv["low"][:-5]
        c = synthetic_ohlcv["close"]
        with pytest.raises(ValueError, match="same length"):
            atr_series(h, l, c)

    def test_high_vol_gives_higher_atr(self):
        n = 100
        low_vol_close  = np.full(n, 100.0) + np.arange(n) * 0.01
        high_vol_close = low_vol_close + np.random.default_rng(7).normal(0, 5, n)
        h_lv = low_vol_close + 0.5
        l_lv = low_vol_close - 0.5
        h_hv = high_vol_close + 5.0
        l_hv = high_vol_close - 5.0
        atr_lv = atr_series(h_lv, l_lv, low_vol_close, period=14)
        atr_hv = atr_series(h_hv, l_hv, high_vol_close, period=14)
        lv_mean = float(np.nanmean(atr_lv))
        hv_mean = float(np.nanmean(atr_hv))
        assert hv_mean > lv_mean * 2  # high vol ATR should be much larger


# ──────────────────── adx_series ────────────────────

class TestAdxSeries:
    def test_output_keys(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        assert set(result.keys()) == {"adx", "plus_di", "minus_di"}

    def test_output_shapes(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        n = len(c)
        for k in ("adx", "plus_di", "minus_di"):
            assert result[k].shape == (n,), f"shape mismatch for {k}"

    def test_adx_range(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        valid = result["adx"][~np.isnan(result["adx"])]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_di_range(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        for key in ("plus_di", "minus_di"):
            v = result[key][~np.isnan(result[key])]
            assert np.all(v >= 0)

    def test_warmup_nans_adx(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        period = 14
        result = adx_series(h, l, c, period=period)
        # ADX needs 2*period warmup bars
        assert np.all(np.isnan(result["adx"][:2 * period]))

    def test_strong_uptrend_plus_di_dominant(self, uptrend_ohlcv):
        h, l, c = uptrend_ohlcv["high"], uptrend_ohlcv["low"], uptrend_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        valid_mask = ~np.isnan(result["plus_di"]) & ~np.isnan(result["minus_di"])
        if valid_mask.sum() > 10:
            pdi = result["plus_di"][valid_mask]
            mdi = result["minus_di"][valid_mask]
            # In a strong uptrend, +DI should dominate on average
            assert float(np.mean(pdi)) > float(np.mean(mdi))

    def test_strong_downtrend_minus_di_dominant(self, downtrend_ohlcv):
        h, l, c = downtrend_ohlcv["high"], downtrend_ohlcv["low"], downtrend_ohlcv["close"]
        result = adx_series(h, l, c, period=14)
        valid_mask = ~np.isnan(result["plus_di"]) & ~np.isnan(result["minus_di"])
        if valid_mask.sum() > 10:
            pdi = result["plus_di"][valid_mask]
            mdi = result["minus_di"][valid_mask]
            assert float(np.mean(mdi)) > float(np.mean(pdi))

    def test_series_input(self, synthetic_ohlcv):
        idx = pd.date_range("2020-01-01", periods=len(synthetic_ohlcv["close"]), freq="1H")
        h = pd.Series(synthetic_ohlcv["high"], index=idx)
        l = pd.Series(synthetic_ohlcv["low"], index=idx)
        c = pd.Series(synthetic_ohlcv["close"], index=idx)
        result = adx_series(h, l, c, period=14)
        assert isinstance(result["adx"], pd.Series)
        assert result["adx"].index.equals(idx)

    def test_invalid_period_raises(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        with pytest.raises(ValueError, match="period"):
            adx_series(h, l, c, period=-1)


# ──────────────────── supertrend ────────────────────

class TestSupertrend:
    def test_output_keys(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = supertrend(h, l, c, period=10, multiplier=3.0)
        assert set(result.keys()) == {"direction", "upper_band", "lower_band", "line"}

    def test_output_shapes(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = supertrend(h, l, c)
        n = len(c)
        for k in result:
            assert result[k].shape == (n,)

    def test_direction_values(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = supertrend(h, l, c)
        valid = result["direction"][~np.isnan(result["direction"])]
        assert set(valid).issubset({1.0, -1.0})

    def test_uptrend_gives_plus1(self, uptrend_ohlcv):
        h, l, c = uptrend_ohlcv["high"], uptrend_ohlcv["low"], uptrend_ohlcv["close"]
        result = supertrend(h, l, c, period=10, multiplier=3.0)
        valid = result["direction"][~np.isnan(result["direction"])]
        # Strong uptrend: most bars should show +1
        assert float(np.mean(valid)) > 0.5

    def test_downtrend_gives_minus1(self, downtrend_ohlcv):
        h, l, c = downtrend_ohlcv["high"], downtrend_ohlcv["low"], downtrend_ohlcv["close"]
        result = supertrend(h, l, c, period=10, multiplier=3.0)
        valid = result["direction"][~np.isnan(result["direction"])]
        assert float(np.mean(valid)) < -0.5

    def test_line_follows_correct_band(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        result = supertrend(h, l, c)
        direction = result["direction"]
        line      = result["line"]
        lb        = result["lower_band"]
        ub        = result["upper_band"]
        for i in range(len(c)):
            if np.isnan(direction[i]) or np.isnan(line[i]):
                continue
            if direction[i] == 1.0:
                assert np.isclose(line[i], lb[i], rtol=1e-9), f"line≠lower_band at {i}"
            else:
                assert np.isclose(line[i], ub[i], rtol=1e-9), f"line≠upper_band at {i}"

    def test_series_input_type_preserved(self, synthetic_ohlcv):
        idx = pd.date_range("2020-01-01", periods=len(synthetic_ohlcv["close"]), freq="1H")
        h = pd.Series(synthetic_ohlcv["high"], index=idx)
        l = pd.Series(synthetic_ohlcv["low"], index=idx)
        c = pd.Series(synthetic_ohlcv["close"], index=idx)
        result = supertrend(h, l, c)
        assert isinstance(result["direction"], pd.Series)
        assert result["direction"].index.equals(idx)

    def test_invalid_multiplier_raises(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        with pytest.raises(ValueError, match="multiplier"):
            supertrend(h, l, c, multiplier=0.0)

    def test_warmup_nans(self, synthetic_ohlcv):
        h, l, c = synthetic_ohlcv["high"], synthetic_ohlcv["low"], synthetic_ohlcv["close"]
        period = 10
        result = supertrend(h, l, c, period=period)
        # First period bars should have NaN direction
        assert np.all(np.isnan(result["direction"][:period]))

    def test_mismatched_lengths_raises(self, synthetic_ohlcv):
        h = synthetic_ohlcv["high"]
        l = synthetic_ohlcv["low"][:-3]
        c = synthetic_ohlcv["close"]
        with pytest.raises(ValueError, match="same length"):
            supertrend(h, l, c)
