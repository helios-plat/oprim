"""Tests for oprim.technical.volume: obv, mfi."""
import numpy as np
import pandas as pd
import pytest

from oprim.technical.volume import mfi, obv


def _make_data(n=100, seed=42):
    rng = np.random.default_rng(seed)
    closes = np.cumprod(1.0 + rng.standard_normal(n) * 0.01) * 100
    highs = closes + np.abs(rng.standard_normal(n)) * 0.5
    lows = closes - np.abs(rng.standard_normal(n)) * 0.5
    volumes = rng.uniform(1e5, 1e6, n)
    return highs, lows, closes, volumes


# ---------- OBV ----------

def test_obv_starts_at_zero():
    _, _, c, v = _make_data()
    result = obv(c, v)
    assert float(np.array(result)[0]) == 0.0


def test_obv_shape():
    _, _, c, v = _make_data(80)
    result = obv(c, v)
    assert len(result) == 80


def test_obv_increases_on_up_day():
    closes = np.array([10.0, 11.0, 11.0])
    volumes = np.array([100.0, 200.0, 150.0])
    result = np.array(obv(closes, volumes))
    # Day 1: up → OBV[1] = 200
    assert result[1] == pytest.approx(200.0)
    # Day 2: no change → OBV[2] = OBV[1]
    assert result[2] == pytest.approx(200.0)


def test_obv_decreases_on_down_day():
    closes = np.array([11.0, 10.0])
    volumes = np.array([100.0, 200.0])
    result = np.array(obv(closes, volumes))
    assert result[1] == pytest.approx(-200.0)


def test_obv_series_input():
    _, _, c, v = _make_data(50)
    result = obv(pd.Series(c), pd.Series(v))
    assert isinstance(result, pd.Series)


def test_obv_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        obv(np.array([]), np.array([]))


def test_obv_length_mismatch_raises():
    _, _, c, v = _make_data(50)
    with pytest.raises(ValueError, match="same length"):
        obv(c, v[:30])


# ---------- MFI ----------

def test_mfi_output_shape():
    h, l, c, v = _make_data(100)
    result = mfi(h, l, c, v)
    assert len(result) == 100


def test_mfi_nan_prefix():
    h, l, c, v = _make_data(100)
    result = mfi(h, l, c, v, period=14)
    arr = np.array(result)
    assert np.all(np.isnan(arr[:14]))
    assert np.isfinite(arr[14])


def test_mfi_normalized_range():
    h, l, c, v = _make_data(100)
    result = mfi(h, l, c, v, normalize=True)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all((valid >= 0) & (valid <= 1))


def test_mfi_unnormalized_range():
    h, l, c, v = _make_data(100)
    result = mfi(h, l, c, v, normalize=False)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all((valid >= 0) & (valid <= 100))


def test_mfi_series_input():
    h, l, c, v = _make_data(50)
    result = mfi(pd.Series(h), pd.Series(l), pd.Series(c), pd.Series(v))
    assert isinstance(result, pd.Series)


def test_mfi_empty_raises():
    with pytest.raises(ValueError):
        mfi(np.array([]), np.array([]), np.array([]), np.array([]))


def test_mfi_invalid_period_raises():
    h, l, c, v = _make_data(50)
    with pytest.raises(ValueError, match="period"):
        mfi(h, l, c, v, period=0)


def test_mfi_length_mismatch_raises():
    h, l, c, v = _make_data(50)
    with pytest.raises(ValueError, match="same length"):
        mfi(h, l[:30], c, v)


def test_mfi_all_positive_flow():
    """All rising prices → neg_mf=0 → MFI=1.0 (normalized) or 100."""
    n = 30
    # Strictly increasing typical prices (ensures no negative flow)
    h = np.arange(1.0, n + 1) + 1.0
    l = np.arange(1.0, n + 1) - 0.5
    c = np.arange(1.0, n + 1)
    v = np.ones(n) * 1000
    result = mfi(h, l, c, v, period=5, normalize=True)
    arr = np.array(result)
    valid = arr[~np.isnan(arr)]
    assert np.all(valid == pytest.approx(1.0))
