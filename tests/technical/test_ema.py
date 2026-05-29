"""Tests for oprim.ema (Exponential Moving Average)."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.moving_averages import ema


def test_ema_basic_adjust_false():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(prices, 3, adjust=False)
    assert len(result) == 5
    assert np.isfinite(result[0])


def test_ema_basic_adjust_true():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ema(prices, 3, adjust=True)
    assert len(result) == 5


def test_ema_first_value_equals_input():
    prices = np.array([10.0, 20.0, 30.0])
    result = ema(prices, 5, adjust=False)
    assert result[0] == pytest.approx(prices[0])


def test_ema_invalid_window():
    with pytest.raises(ValueError):
        ema(np.array([1.0, 2.0, 3.0]), 0)
    with pytest.raises(ValueError):
        ema(np.array([1.0, 2.0, 3.0]), -1)


def test_ema_empty_input():
    with pytest.raises(ValueError):
        ema(np.array([]), 3)


def test_ema_preserves_series_index():
    idx = pd.date_range("2024-01-01", periods=5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    result = ema(s, 3)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(idx)


def test_ema_ndarray_out():
    arr = np.arange(1.0, 6.0)
    result = ema(arr, 3)
    assert isinstance(result, np.ndarray)


def test_ema_monotone_uptrend():
    prices = np.linspace(10, 20, 50)
    result_f = ema(prices, 10, adjust=False)
    # EMA should be increasing for monotone series
    valid = result_f[np.isfinite(result_f)]
    assert np.all(np.diff(valid) >= 0)


@pytest.mark.academic_reference
def test_ema_matches_pandas_ewm_adjust_false():
    """EMA (adjust=False) must match pd.ewm(span, adjust=False).mean(), rtol=1e-12."""
    rng = np.random.default_rng(1)
    prices = pd.Series(rng.uniform(100, 200, size=100))
    for window in [3, 5, 12, 26]:
        expected = prices.ewm(span=window, adjust=False).mean()
        result = ema(prices, window, adjust=False)
        np.testing.assert_allclose(
            np.asarray(result),
            expected.to_numpy(),
            rtol=1e-12,
            err_msg=f"window={window}, adjust=False",
        )


@pytest.mark.academic_reference
def test_ema_matches_pandas_ewm_adjust_true():
    """EMA (adjust=True) must match pd.ewm(span, adjust=True).mean(), rtol=1e-12."""
    rng = np.random.default_rng(2)
    prices = pd.Series(rng.uniform(100, 200, size=100))
    for window in [3, 5, 12]:
        expected = prices.ewm(span=window, adjust=True).mean()
        result = ema(prices, window, adjust=True)
        np.testing.assert_allclose(
            np.asarray(result),
            expected.to_numpy(),
            rtol=1e-12,
            err_msg=f"window={window}, adjust=True",
        )


def test_ema_nan_matches_pandas():
    """EMA NaN behavior must match pandas ewm exactly."""
    prices = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0])
    result = ema(prices, 3, adjust=False)
    expected = prices.ewm(span=3, adjust=False).mean()
    np.testing.assert_allclose(np.asarray(result), expected.to_numpy(), rtol=1e-12, equal_nan=True)
