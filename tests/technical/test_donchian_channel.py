"""Tests for oprim.donchian_channel."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.bands import donchian_channel


def test_donchian_basic():
    highs = np.array([10.0, 12.0, 11.0, 14.0, 13.0])
    lows = np.array([8.0, 9.0, 7.0, 10.0, 9.0])
    result = donchian_channel(highs, lows, window=3)
    assert set(result.keys()) == {"upper", "middle", "lower"}
    # t=2: max(10,12,11)=12, min(8,9,7)=7
    assert result["upper"][2] == pytest.approx(12.0)
    assert result["lower"][2] == pytest.approx(7.0)
    assert result["middle"][2] == pytest.approx(9.5)


def test_donchian_constant_prices():
    prices = np.full(10, 50.0)
    result = donchian_channel(prices, prices, window=5)
    valid = ~np.isnan(result["upper"])
    assert np.all(result["upper"][valid] == pytest.approx(50.0))
    assert np.all(result["lower"][valid] == pytest.approx(50.0))
    assert np.all(result["middle"][valid] == pytest.approx(50.0))


def test_donchian_strict_uptrend():
    highs = np.arange(1.0, 11.0)
    lows = np.arange(0.5, 10.5)
    result = donchian_channel(highs, lows, window=3)
    # Upper should be the highest high in each window
    t = 9
    assert result["upper"][t] == pytest.approx(max(highs[7:10]))


def test_donchian_mismatched_length_raises():
    with pytest.raises(ValueError):
        donchian_channel(np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5]))


def test_donchian_invalid_window():
    with pytest.raises(ValueError):
        donchian_channel(np.arange(1.0, 10.0), np.arange(0.5, 9.5), window=0)
    with pytest.raises(ValueError):
        donchian_channel(np.arange(1.0, 10.0), np.arange(0.5, 9.5), window=-1)


def test_donchian_nan_before_window():
    highs = np.arange(1.0, 11.0)
    lows = np.arange(0.5, 10.5)
    result = donchian_channel(highs, lows, window=4)
    assert all(np.isnan(result["upper"][:3]))
    assert np.isfinite(result["upper"][3])


def test_donchian_preserves_series():
    idx = pd.date_range("2024-01-01", periods=10)
    highs = pd.Series(np.arange(1.0, 11.0), index=idx)
    lows = pd.Series(np.arange(0.5, 10.5), index=idx)
    result = donchian_channel(highs, lows, window=3)
    assert isinstance(result["upper"], pd.Series)
    assert list(result["upper"].index) == list(idx)


@pytest.mark.academic_reference
def test_donchian_matches_manual_rolling_max_min():
    """Donchian Channel must match pd.rolling().max() / .min(), rtol=1e-12."""
    rng = np.random.default_rng(3)
    h_base = rng.uniform(100, 200, 100)
    l_base = h_base - rng.uniform(0, 5, 100)
    highs = pd.Series(h_base)
    lows = pd.Series(l_base)
    window = 20

    expected_upper = highs.rolling(window).max()
    expected_lower = lows.rolling(window).min()

    result = donchian_channel(highs, lows, window=window)
    mask = expected_upper.notna()

    np.testing.assert_allclose(
        np.asarray(result["upper"])[mask],
        expected_upper[mask].to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(result["lower"])[mask],
        expected_lower[mask].to_numpy(),
        rtol=1e-12,
    )
