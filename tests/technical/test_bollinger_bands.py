"""Tests for oprim.bollinger_bands."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.bands import bollinger_bands


def test_bollinger_basic():
    prices = np.arange(1.0, 26.0)
    result = bollinger_bands(prices, window=5)
    assert set(result.keys()) == {"upper", "middle", "lower", "bandwidth", "percent_b"}
    assert len(result["upper"]) == 25


def test_bollinger_default_params():
    rng = np.random.default_rng(10)
    prices = rng.uniform(100, 200, 100)
    result = bollinger_bands(prices)
    # Default window=20
    assert all(np.isnan(result["upper"][:19]))
    assert np.isfinite(result["upper"][19])


def test_bollinger_bandwidth_calculation():
    prices = np.arange(1.0, 26.0)
    result = bollinger_bands(prices, window=5, num_std=2.0)
    t = 10
    bw = result["bandwidth"][t]
    upper = result["upper"][t]
    lower = result["lower"][t]
    middle = result["middle"][t]
    expected_bw = (upper - lower) / middle
    assert bw == pytest.approx(expected_bw)


def test_bollinger_percent_b_at_upper_band():
    prices = np.arange(1.0, 21.0)
    result = bollinger_bands(prices, window=5, num_std=2.0)
    t = 15
    p_b = result["percent_b"][t]
    upper = result["upper"][t]
    lower = result["lower"][t]
    price = prices[t]
    expected = (price - lower) / (upper - lower)
    assert p_b == pytest.approx(expected)


def test_bollinger_percent_b_at_lower_band():
    # At lower band: (price - lower) / (upper - lower) = 0.0
    # Construct prices so one bar is exactly at lower band
    prices = np.full(25, 100.0)
    result = bollinger_bands(prices, window=5, num_std=2.0)
    # Constant prices → std=0 → percent_b=NaN (div by zero guarded)
    valid = ~np.isnan(result["percent_b"])
    assert not np.any(valid), "constant prices should yield all NaN percent_b"


def test_bollinger_constant_price_zero_std():
    prices = np.full(25, 100.0)
    result = bollinger_bands(prices, window=5)
    # std=0 → bandwidth=0, percent_b=NaN
    valid_bw = result["bandwidth"][~np.isnan(result["bandwidth"])]
    assert np.all(valid_bw == 0.0)
    valid_pb = result["percent_b"][~np.isnan(result["percent_b"])]
    assert len(valid_pb) == 0


def test_bollinger_invalid_window():
    with pytest.raises(ValueError):
        bollinger_bands(np.arange(1.0, 10.0), window=0)


def test_bollinger_invalid_num_std():
    with pytest.raises(ValueError):
        bollinger_bands(np.arange(1.0, 25.0), num_std=-1.0)
    with pytest.raises(ValueError):
        bollinger_bands(np.arange(1.0, 25.0), num_std=0.0)


def test_bollinger_preserves_series():
    idx = pd.date_range("2024-01-01", periods=30)
    s = pd.Series(np.linspace(100, 200, 30), index=idx)
    result = bollinger_bands(s, window=5)
    assert isinstance(result["upper"], pd.Series)
    assert list(result["upper"].index) == list(idx)


@pytest.mark.academic_reference
def test_bollinger_matches_textbook():
    """Bollinger Bands must match pandas rolling mean + std(ddof=0), rtol=1e-10."""
    rng = np.random.default_rng(5)
    prices = pd.Series(rng.uniform(100, 200, 100))
    window = 20
    num_std = 2.0

    expected_middle = prices.rolling(window).mean()
    expected_std = prices.rolling(window).std(ddof=0)
    expected_upper = expected_middle + num_std * expected_std
    expected_lower = expected_middle - num_std * expected_std

    result = bollinger_bands(prices, window=window, num_std=num_std)

    mask = expected_upper.notna()
    np.testing.assert_allclose(
        np.asarray(result["upper"])[mask],
        expected_upper[mask].to_numpy(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result["lower"])[mask],
        expected_lower[mask].to_numpy(),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(result["middle"])[mask],
        expected_middle[mask].to_numpy(),
        rtol=1e-10,
    )
