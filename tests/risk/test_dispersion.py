"""Tests for oprim.risk.dispersion.mean_deviation."""
import numpy as np
import pandas as pd
import pytest

from oprim.risk.dispersion import mean_deviation


def test_scalar_constant_series():
    x = np.ones(10)
    assert mean_deviation(x) == pytest.approx(0.0)


def test_scalar_simple():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # mean=3, deviations=[2,1,0,1,2], mean_dev=1.2
    assert mean_deviation(x) == pytest.approx(1.2)


def test_scalar_median_center():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # median=3, same deviations
    assert mean_deviation(x, center="median") == pytest.approx(1.2)


def test_rolling_mean_shape():
    x = np.arange(20.0)
    result = mean_deviation(x, window=5)
    assert result.shape == (20,)
    assert np.all(np.isnan(result[:4]))
    assert np.all(np.isfinite(result[4:]))


def test_rolling_median_shape():
    x = np.arange(20.0)
    result = mean_deviation(x, window=5, center="median")
    assert result.shape == (20,)


def test_zero_variance_rolling():
    x = np.ones(10)
    result = mean_deviation(x, window=3)
    assert np.allclose(result[2:], 0.0)


def test_returns_series_for_series_input():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mean_deviation(s, window=3)
    assert isinstance(result, pd.Series)


def test_bad_center_raises():
    with pytest.raises(ValueError, match="center"):
        mean_deviation(np.array([1.0, 2.0]), center="mode")


def test_nonpositive_window_raises():
    with pytest.raises(ValueError, match="window"):
        mean_deviation(np.array([1.0, 2.0, 3.0]), window=0)


def test_rolling_values_correct():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mean_deviation(x, window=3)
    # window=3, pos 2: [1,2,3], mean=2, devs=[1,0,1], md=2/3
    assert result[2] == pytest.approx(2.0 / 3.0)
    # pos 3: [2,3,4], mean=3, devs=[1,0,1], md=2/3
    assert result[3] == pytest.approx(2.0 / 3.0)
