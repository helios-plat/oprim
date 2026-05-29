"""Tests for oprim.macd (MACD indicator)."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.moving_averages import macd


def test_macd_default_params():
    rng = np.random.default_rng(42)
    prices = rng.uniform(100, 200, size=100)
    result = macd(prices)
    assert set(result.keys()) == {"macd", "signal", "histogram"}
    assert len(result["macd"]) == 100


def test_macd_custom_params():
    prices = np.linspace(100, 150, 50)
    result = macd(prices, fast_period=5, slow_period=10, signal_period=3)
    assert len(result["macd"]) == 50


def test_macd_fast_geq_slow_raises():
    with pytest.raises(ValueError):
        macd(np.arange(50.0), fast_period=26, slow_period=12)
    with pytest.raises(ValueError):
        macd(np.arange(50.0), fast_period=12, slow_period=12)


def test_macd_returns_dict_with_three_keys():
    result = macd(np.linspace(1, 10, 50))
    assert "macd" in result
    assert "signal" in result
    assert "histogram" in result


def test_macd_histogram_equals_macd_minus_signal():
    prices = np.linspace(100, 200, 60)
    result = macd(prices, fast_period=5, slow_period=10, signal_period=3)
    diff = result["macd"] - result["signal"]
    np.testing.assert_allclose(result["histogram"], diff, rtol=1e-12)


def test_macd_invalid_period_raises():
    with pytest.raises(ValueError):
        macd(np.arange(50.0), fast_period=0, slow_period=12)
    with pytest.raises(ValueError):
        macd(np.arange(50.0), fast_period=5, slow_period=-1)


def test_macd_preserves_series_type():
    idx = pd.date_range("2024-01-01", periods=60)
    s = pd.Series(np.linspace(100, 200, 60), index=idx)
    result = macd(s, fast_period=5, slow_period=10, signal_period=3)
    assert isinstance(result["macd"], pd.Series)
    assert list(result["macd"].index) == list(idx)


@pytest.mark.academic_reference
def test_macd_matches_talib():
    """MACD must match pandas EWM manual implementation, rtol=1e-8."""
    rng = np.random.default_rng(0)
    prices = pd.Series(rng.uniform(100, 200, size=200))

    fast, slow, signal = 12, 26, 9
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    expected_hist = macd_line - signal_line

    result = macd(prices, fast_period=fast, slow_period=slow, signal_period=signal)

    np.testing.assert_allclose(
        np.asarray(result["macd"]),
        macd_line.to_numpy(),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result["signal"]),
        signal_line.to_numpy(),
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result["histogram"]),
        expected_hist.to_numpy(),
        rtol=1e-8,
    )


def test_macd_nan_first_value():
    """_ema_recursive: NaN in first price propagates through MACD."""
    prices = np.array([np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
    result = macd(prices, fast_period=3, slow_period=5, signal_period=2)
    # First value NaN propagates
    assert np.isnan(result["macd"][0])


def test_macd_nan_mid_series():
    """_ema_recursive: NaN in mid-series propagates."""
    prices = np.arange(1.0, 31.0)
    prices[5] = np.nan
    result = macd(prices, fast_period=3, slow_period=5, signal_period=2)
    assert np.isnan(result["macd"][5])
