"""Tests for oprim.vwap (Volume Weighted Average Price)."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.moving_averages import vwap


def test_vwap_cumulative_basic():
    prices = np.array([10.0, 20.0, 30.0])
    volumes = np.array([1.0, 2.0, 3.0])
    result = vwap(prices, volumes)
    # Cumulative VWAP at t=2: (10*1 + 20*2 + 30*3) / (1+2+3) = 140/6
    assert result[2] == pytest.approx(140.0 / 6.0)


def test_vwap_rolling_basic():
    prices = np.array([10.0, 20.0, 30.0, 40.0])
    volumes = np.array([1.0, 1.0, 1.0, 1.0])
    result = vwap(prices, volumes, window=2)
    assert np.isnan(result[0])
    assert result[1] == pytest.approx(15.0)
    assert result[2] == pytest.approx(25.0)
    assert result[3] == pytest.approx(35.0)


def test_vwap_zero_volume_returns_nan():
    prices = np.array([10.0, 20.0, 30.0])
    volumes = np.array([0.0, 0.0, 0.0])
    result = vwap(prices, volumes)
    assert all(np.isnan(result))


def test_vwap_negative_volume_raises():
    with pytest.raises(ValueError):
        vwap(np.array([1.0, 2.0]), np.array([1.0, -1.0]))


def test_vwap_mismatched_length_raises():
    with pytest.raises(ValueError):
        vwap(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_vwap_window_equals_length():
    prices = np.array([10.0, 20.0, 30.0])
    volumes = np.array([2.0, 3.0, 5.0])
    result = vwap(prices, volumes, window=3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    expected = (10*2 + 20*3 + 30*5) / 10
    assert result[2] == pytest.approx(expected)


def test_vwap_preserves_series_type():
    idx = pd.date_range("2024-01-01", periods=4)
    p = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    v = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    result = vwap(p, v)
    assert isinstance(result, pd.Series)


def test_vwap_invalid_window():
    with pytest.raises(ValueError):
        vwap(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), window=0)


@pytest.mark.academic_reference
def test_vwap_matches_manual_calculation():
    """VWAP must match manually computed reference values, rtol=1e-12."""
    prices = np.array([100.0, 101.0, 99.5, 102.0, 100.5])
    volumes = np.array([500.0, 800.0, 300.0, 1000.0, 600.0])
    # Cumulative VWAP at each step
    pv = prices * volumes
    cv = np.cumsum(volumes)
    cpv = np.cumsum(pv)
    expected = cpv / cv
    result = vwap(prices, volumes)
    np.testing.assert_allclose(result, expected, rtol=1e-12)
