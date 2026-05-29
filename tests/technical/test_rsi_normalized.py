"""Tests for oprim.rsi_normalized."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.oscillators import rsi_normalized


def test_rsi_normalized_basic_uptrend():
    prices = np.linspace(10, 100, 50)
    result = rsi_normalized(prices, period=14)
    valid = result[~np.isnan(result)]
    # All gains → RSI approaches 1
    assert np.all(valid > 0.9)


def test_rsi_normalized_basic_downtrend():
    prices = np.linspace(100, 10, 50)
    result = rsi_normalized(prices, period=14)
    valid = result[~np.isnan(result)]
    # All losses → RSI approaches 0
    assert np.all(valid < 0.1)


def test_rsi_normalized_invalid_period():
    with pytest.raises(ValueError):
        rsi_normalized(np.arange(1.0, 20.0), period=0)
    with pytest.raises(ValueError):
        rsi_normalized(np.arange(1.0, 20.0), period=-5)


def test_rsi_normalized_empty_input():
    with pytest.raises(ValueError):
        rsi_normalized(np.array([]))


def test_rsi_normalized_zero_loss_returns_one():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                       12.0, 13.0, 14.0, 15.0, 16.0])
    result = rsi_normalized(prices, period=5)
    valid = result[np.isfinite(result)]
    assert np.all(valid == pytest.approx(1.0, abs=1e-10))


def test_rsi_normalized_in_zero_one_range():
    rng = np.random.default_rng(99)
    prices = rng.uniform(100, 200, 100)
    result = rsi_normalized(prices, period=14)
    valid = result[np.isfinite(result)]
    assert np.all(valid >= 0.0) and np.all(valid <= 1.0)


def test_rsi_normalized_nan_for_first_period():
    prices = np.arange(1.0, 30.0)
    period = 14
    result = rsi_normalized(prices, period=period)
    assert all(np.isnan(result[:period]))
    assert np.isfinite(result[period])


def test_rsi_normalized_preserves_series():
    idx = pd.date_range("2024-01-01", periods=30)
    s = pd.Series(np.linspace(100, 200, 30), index=idx)
    result = rsi_normalized(s, period=10)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(idx)


@pytest.mark.academic_reference
def test_rsi_normalized_matches_wilder_definition():
    """RSI normalized must match Wilder (1978) SMA-seeded smoothing definition."""
    rng = np.random.default_rng(7)
    prices_arr = rng.uniform(100, 200, 200)
    period = 14

    # Reference: manual Wilder smoothing (SMA seed, then EMA with alpha=1/period)
    deltas = np.diff(prices_arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    expected = np.full(len(prices_arr), np.nan)
    avg_g = gains[:period].mean()
    avg_l = losses[:period].mean()
    expected[period] = 1.0 if avg_l == 0 else 1.0 - 1.0 / (1.0 + avg_g / avg_l)
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        expected[i + 1] = 1.0 if avg_l == 0 else 1.0 - 1.0 / (1.0 + avg_g / avg_l)

    result = rsi_normalized(prices_arr, period=period)
    valid = np.isfinite(expected) & np.isfinite(result)
    assert valid.sum() > 100
    np.testing.assert_allclose(result[valid], expected[valid], rtol=1e-12)
