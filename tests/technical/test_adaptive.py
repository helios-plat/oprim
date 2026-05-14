"""Tests for oprim.technical.adaptive: kama."""
import numpy as np
import pandas as pd
import pytest

from oprim.technical.adaptive import kama


def _linear_trend(n=100):
    return np.arange(1.0, n + 1)


def _random_walk(n=100, seed=42):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal(n))


def test_kama_output_shape():
    prices = _random_walk(100)
    result = kama(prices)
    assert len(result) == 100


def test_kama_nan_prefix():
    prices = _random_walk(100)
    result = kama(prices, er_period=10)
    assert np.all(np.isnan(result[:10]))
    assert np.isfinite(result[10])


def test_kama_returns_series_for_series_input():
    s = pd.Series(_random_walk(100))
    result = kama(s)
    assert isinstance(result, pd.Series)


def test_kama_trending_follows_closely():
    """In a strong trend, ER=1, KAMA should use fast EMA."""
    prices = _linear_trend(50)
    result = kama(prices, er_period=5, fast_period=2, slow_period=10)
    valid = result[~np.isnan(result)]
    # In perfect trend, KAMA converges close to prices
    assert valid[-1] == pytest.approx(prices[-1], rel=0.05)


def test_kama_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        kama(np.array([]))


def test_kama_invalid_er_period_raises():
    with pytest.raises(ValueError, match="er_period"):
        kama(np.arange(50.0), er_period=0)


def test_kama_slow_period_must_exceed_fast():
    with pytest.raises(ValueError, match="slow_period"):
        kama(np.arange(50.0), fast_period=10, slow_period=5)


def test_kama_invalid_fast_period_raises():
    with pytest.raises(ValueError, match="fast_period"):
        kama(np.arange(50.0), fast_period=0)


def test_kama_short_series_returns_all_nan():
    prices = np.arange(5.0)  # shorter than default er_period=10
    result = kama(prices, er_period=10)
    assert np.all(np.isnan(result))
