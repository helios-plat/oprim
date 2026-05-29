"""Tests for oprim.technical.bands: keltner_channels."""
import numpy as np
import pandas as pd
import pytest

from oprim.technical.bands import keltner_channels


def _make_ohlc(n=100, seed=42):
    rng = np.random.default_rng(seed)
    closes = np.cumprod(1.0 + rng.standard_normal(n) * 0.01) * 100
    highs = closes + np.abs(rng.standard_normal(n)) * 0.5
    lows = closes - np.abs(rng.standard_normal(n)) * 0.5
    return highs, lows, closes


def test_keltner_output_keys():
    h, l, c = _make_ohlc()
    r = keltner_channels(h, l, c)
    assert "upper" in r and "middle" in r and "lower" in r


def test_keltner_upper_above_lower():
    h, l, c = _make_ohlc()
    r = keltner_channels(h, l, c)
    upper = np.array(r["upper"])
    lower = np.array(r["lower"])
    valid = ~np.isnan(upper) & ~np.isnan(lower)
    assert np.all(upper[valid] > lower[valid])


def test_keltner_shape():
    h, l, c = _make_ohlc(80)
    r = keltner_channels(h, l, c)
    assert len(r["middle"]) == 80


def test_keltner_series_input():
    h, l, c = _make_ohlc(80)
    r = keltner_channels(pd.Series(h), pd.Series(l), pd.Series(c))
    assert isinstance(r["middle"], pd.Series)


def test_keltner_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        keltner_channels(np.array([]), np.array([]), np.array([]))


def test_keltner_invalid_period_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="ema_period"):
        keltner_channels(h, l, c, ema_period=0)


def test_keltner_length_mismatch_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="same length"):
        keltner_channels(h, l[:30], c)


def test_keltner_invalid_multiplier_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError, match="multiplier"):
        keltner_channels(h, l, c, multiplier=0.0)
