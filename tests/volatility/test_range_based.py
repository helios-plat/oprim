"""Tests for oprim.volatility.range_based: Parkinson, Garman-Klass, Yang-Zhang."""
import numpy as np
import pytest

from oprim.volatility.range_based import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)


def _make_ohlc(n=100, sigma=0.01, seed=42):
    """Generate synthetic OHLC data."""
    rng = np.random.default_rng(seed)
    closes = np.cumprod(1.0 + rng.standard_normal(n) * sigma) * 100.0
    highs = closes * (1 + np.abs(rng.standard_normal(n)) * sigma)
    lows = closes * (1 - np.abs(rng.standard_normal(n)) * sigma)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return opens, highs, lows, closes


# ---------- Parkinson ----------

def test_parkinson_positive():
    _, h, l, _ = _make_ohlc()
    sigma = parkinson_volatility(h, l)
    assert sigma > 0


def test_parkinson_float_output():
    _, h, l, _ = _make_ohlc()
    sigma = parkinson_volatility(h, l)
    assert isinstance(sigma, float)


def test_parkinson_annualize():
    _, h, l, _ = _make_ohlc(100)
    sigma_d = parkinson_volatility(h, l)
    sigma_a = parkinson_volatility(h, l, annualize=True)
    assert sigma_a == pytest.approx(sigma_d * np.sqrt(252))


def test_parkinson_constant_hl_zero():
    h = np.ones(50) * 10.0
    l = np.ones(50) * 10.0
    sigma = parkinson_volatility(h, l)
    assert sigma == pytest.approx(0.0)


def test_parkinson_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parkinson_volatility(np.array([]), np.array([]))


def test_parkinson_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        parkinson_volatility(np.ones(10), np.ones(5))


def test_parkinson_high_less_than_low_raises():
    with pytest.raises(ValueError, match="highs must be"):
        parkinson_volatility(np.array([1.0, 0.9]), np.array([1.0, 1.0]))


# ---------- Garman-Klass ----------

def test_garman_klass_positive():
    o, h, l, c = _make_ohlc()
    sigma = garman_klass_volatility(o, h, l, c)
    assert sigma > 0


def test_garman_klass_float_output():
    o, h, l, c = _make_ohlc()
    sigma = garman_klass_volatility(o, h, l, c)
    assert isinstance(sigma, float)


def test_garman_klass_annualize():
    o, h, l, c = _make_ohlc()
    sigma_d = garman_klass_volatility(o, h, l, c)
    sigma_a = garman_klass_volatility(o, h, l, c, annualize=True)
    assert sigma_a == pytest.approx(sigma_d * np.sqrt(252))


def test_garman_klass_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        garman_klass_volatility(
            np.array([]), np.array([]), np.array([]), np.array([])
        )


def test_garman_klass_length_mismatch_raises():
    o, h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="same length"):
        garman_klass_volatility(o, h[:30], l, c)


def test_garman_klass_more_efficient_than_cc():
    """GK uses more information than close-to-close; a smoke test."""
    rng = np.random.default_rng(7)
    n = 200
    log_r = rng.standard_normal(n) * 0.01
    c = np.cumprod(np.exp(log_r)) * 100
    h = c * (1 + np.abs(rng.standard_normal(n)) * 0.005)
    l = c * (1 - np.abs(rng.standard_normal(n)) * 0.005)
    o = np.roll(c, 1); o[0] = c[0]
    sigma_gk = garman_klass_volatility(o, h, l, c)
    assert sigma_gk > 0


# ---------- Yang-Zhang ----------

def test_yang_zhang_output_shape():
    o, h, l, c = _make_ohlc(100)
    result = yang_zhang_volatility(o, h, l, c, window=20)
    assert len(result) == 100


def test_yang_zhang_nan_prefix():
    o, h, l, c = _make_ohlc(100)
    result = yang_zhang_volatility(o, h, l, c, window=20)
    assert np.all(np.isnan(result[:20]))
    assert np.all(np.isfinite(result[20:]))


def test_yang_zhang_positive_values():
    o, h, l, c = _make_ohlc(100)
    result = yang_zhang_volatility(o, h, l, c, window=20)
    assert np.all(result[20:] >= 0)


def test_yang_zhang_annualize():
    o, h, l, c = _make_ohlc(100)
    r_d = yang_zhang_volatility(o, h, l, c, window=20)
    r_a = yang_zhang_volatility(o, h, l, c, window=20, annualize=True)
    valid = ~np.isnan(r_d)
    np.testing.assert_allclose(r_a[valid], r_d[valid] * np.sqrt(252))


def test_yang_zhang_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        yang_zhang_volatility(
            np.array([]), np.array([]), np.array([]), np.array([])
        )


def test_yang_zhang_invalid_window_raises():
    o, h, l, c = _make_ohlc(100)
    with pytest.raises(ValueError, match="window"):
        yang_zhang_volatility(o, h, l, c, window=1)


def test_yang_zhang_length_mismatch_raises():
    o, h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="same length"):
        yang_zhang_volatility(o, h[:30], l, c)
