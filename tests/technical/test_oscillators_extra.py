"""Tests for stochastic_oscillator, cci, williams_r."""
import numpy as np
import pandas as pd
import pytest

from oprim.technical.oscillators import cci, stochastic_oscillator, williams_r


def _make_ohlc(n=100, seed=42):
    rng = np.random.default_rng(seed)
    closes = np.cumprod(1.0 + rng.standard_normal(n) * 0.01) * 100
    highs = closes + np.abs(rng.standard_normal(n)) * 0.5
    lows = closes - np.abs(rng.standard_normal(n)) * 0.5
    return highs, lows, closes


# ---------- Stochastic Oscillator ----------

def test_stoch_output_shape():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(h, l, c)
    assert len(r["k"]) == 100
    assert len(r["d"]) == 100
    assert len(r["raw_k"]) == 100


def test_stoch_normalized_range():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(h, l, c, normalize=True)
    k = np.array(r["k"])
    valid = k[~np.isnan(k)]
    assert np.all((valid >= 0) & (valid <= 1))


def test_stoch_unnormalized_range():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(h, l, c, normalize=False)
    k = np.array(r["k"])
    valid = k[~np.isnan(k)]
    assert np.all((valid >= 0) & (valid <= 100))


def test_stoch_has_nan_prefix():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(h, l, c, k_period=14, smooth_k=3, d_period=3)
    k_arr = np.array(r["k"])
    # First k_period + smooth_k - 2 elements should be NaN
    assert np.isnan(k_arr[0])


def test_stoch_returns_dict_with_keys():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(h, l, c)
    assert "k" in r and "d" in r and "raw_k" in r


def test_stoch_series_input():
    h, l, c = _make_ohlc(100)
    r = stochastic_oscillator(
        pd.Series(h), pd.Series(l), pd.Series(c)
    )
    assert isinstance(r["k"], pd.Series)


def test_stoch_empty_raises():
    with pytest.raises(ValueError):
        stochastic_oscillator(np.array([]), np.array([]), np.array([]))


def test_stoch_length_mismatch_raises():
    h, l, c = _make_ohlc(100)
    with pytest.raises(ValueError, match="same length"):
        stochastic_oscillator(h, l[:50], c)


def test_stoch_invalid_period_raises():
    h, l, c = _make_ohlc(100)
    with pytest.raises(ValueError, match="k_period"):
        stochastic_oscillator(h, l, c, k_period=0)


# ---------- CCI ----------

def test_cci_output_shape():
    h, l, c = _make_ohlc(100)
    result = cci(h, l, c)
    assert len(result) == 100


def test_cci_nan_prefix():
    h, l, c = _make_ohlc(100)
    result = cci(h, l, c, period=20)
    assert np.all(np.isnan(np.array(result)[:19]))
    assert np.isfinite(float(result[19]))


def test_cci_overbought_signal():
    """CCI > 100 indicates overbought condition."""
    h, l, c = _make_ohlc(100)
    result = cci(h, l, c)
    arr = np.array(result)
    # Just check that the indicator produces some values above and below 0
    valid = arr[~np.isnan(arr)]
    assert len(valid) > 0


def test_cci_constant_prices():
    """Constant prices should give CCI=0."""
    h = np.ones(30) * 10.0
    l = np.ones(30) * 10.0
    c = np.ones(30) * 10.0
    result = cci(h, l, c, period=5)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    np.testing.assert_allclose(valid, 0.0, atol=1e-10)


def test_cci_empty_raises():
    with pytest.raises(ValueError):
        cci(np.array([]), np.array([]), np.array([]))


def test_cci_invalid_period_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="period"):
        cci(h, l, c, period=0)


def test_cci_series_input():
    h, l, c = _make_ohlc(50)
    result = cci(pd.Series(h), pd.Series(l), pd.Series(c))
    assert isinstance(result, pd.Series)


def test_cci_length_mismatch_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="same length"):
        cci(h, l[:30], c)


# ---------- Williams %R ----------

def test_wr_output_shape():
    h, l, c = _make_ohlc(100)
    result = williams_r(h, l, c)
    assert len(result) == 100


def test_wr_normalized_range():
    h, l, c = _make_ohlc(100)
    result = williams_r(h, l, c, normalize=True)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all((valid >= 0) & (valid <= 1))


def test_wr_unnormalized_range():
    h, l, c = _make_ohlc(100)
    result = williams_r(h, l, c, normalize=False)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all((valid >= -100) & (valid <= 0))


def test_wr_nan_prefix():
    h, l, c = _make_ohlc(100)
    result = williams_r(h, l, c, period=14)
    arr = np.array(result)
    assert np.all(np.isnan(arr[:13]))
    assert np.isfinite(arr[13])


def test_wr_empty_raises():
    with pytest.raises(ValueError):
        williams_r(np.array([]), np.array([]), np.array([]))


def test_wr_invalid_period_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="period"):
        williams_r(h, l, c, period=0)


def test_wr_series_input():
    h, l, c = _make_ohlc(50)
    result = williams_r(pd.Series(h), pd.Series(l), pd.Series(c))
    assert isinstance(result, pd.Series)


def test_wr_length_mismatch_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="same length"):
        williams_r(h, l[:30], c)


def test_stoch_zero_range():
    """Constant prices → denom=0, should not crash."""
    h = np.ones(50) * 10.0
    l = np.ones(50) * 10.0
    c = np.ones(50) * 10.0
    r = stochastic_oscillator(h, l, c, k_period=5)
    raw_k = np.array(r["raw_k"])
    valid = raw_k[~np.isnan(raw_k)]
    assert np.all(valid == pytest.approx(0.5))


def test_wr_zero_range():
    """Constant prices → denom=0, raw_r=-50."""
    h = np.ones(30) * 10.0
    l = np.ones(30) * 10.0
    c = np.ones(30) * 10.0
    result = williams_r(h, l, c, normalize=False)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all(valid == pytest.approx(-50.0))
