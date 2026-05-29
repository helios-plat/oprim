"""Tests for oprim.sma (Simple Moving Average)."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.moving_averages import sma


def test_sma_basic():
    prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    result = sma(prices, 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(2.0)
    assert result[4] == pytest.approx(4.0)
    assert result[9] == pytest.approx(9.0)


def test_sma_full_window_only():
    prices = np.arange(1.0, 6.0)
    result = sma(prices, 3)
    assert all(np.isnan(result[:2]))
    assert all(np.isfinite(result[2:]))


def test_sma_window_equals_length():
    prices = np.array([2.0, 4.0, 6.0])
    result = sma(prices, 3)
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == pytest.approx(4.0)


def test_sma_invalid_window_zero():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0, 3.0]), 0)


def test_sma_invalid_window_negative():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0, 3.0]), -1)


def test_sma_invalid_window_too_large():
    with pytest.raises(ValueError):
        sma(np.array([1.0, 2.0]), 5)


def test_sma_empty_input():
    with pytest.raises(ValueError):
        sma(np.array([]), 3)


def test_sma_preserves_series_index():
    idx = pd.date_range("2024-01-01", periods=5)
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
    result = sma(s, 3)
    assert isinstance(result, pd.Series)
    assert list(result.index) == list(idx)


def test_sma_ndarray_in_ndarray_out():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(arr, 2)
    assert isinstance(result, np.ndarray)


def test_sma_window_one():
    prices = np.array([3.0, 5.0, 7.0])
    result = sma(prices, 1)
    np.testing.assert_array_equal(result, prices)


@pytest.mark.academic_reference
def test_sma_matches_pandas_rolling():
    """SMA must match pd.Series.rolling(window).mean() with rtol=1e-12."""
    rng = np.random.default_rng(0)
    prices = pd.Series(rng.uniform(100, 200, size=100))
    for window in [3, 5, 10, 20]:
        expected = prices.rolling(window).mean()
        result = sma(prices, window)
        mask = expected.notna()
        np.testing.assert_allclose(
            result[mask].to_numpy(),
            expected[mask].to_numpy(),
            rtol=1e-12,
            err_msg=f"window={window}",
        )
