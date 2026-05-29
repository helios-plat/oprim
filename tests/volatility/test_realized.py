"""Tests for oprim.volatility.realized: realized_variance."""
import numpy as np
import pytest

from oprim.volatility.realized import realized_variance


def test_1d_scalar_output():
    r = np.random.default_rng(0).standard_normal(78)
    rv = realized_variance(r)
    assert isinstance(rv, float)
    assert rv == pytest.approx(float(np.sum(r**2)))


def test_2d_array_output():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((10, 78))
    rv = realized_variance(data)
    assert len(rv) == 10
    for i in range(10):
        assert rv[i] == pytest.approx(float(np.sum(data[i] ** 2)))


def test_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        realized_variance(np.array([]))


def test_annualize():
    r = np.random.default_rng(2).standard_normal(78)
    rv_daily = realized_variance(r)
    rv_annual = realized_variance(r, annualize=True, periods_per_year=252)
    assert rv_annual == pytest.approx(rv_daily * 252)


def test_3d_raises():
    with pytest.raises(ValueError, match="1-D or 2-D"):
        realized_variance(np.zeros((5, 10, 3)))


def test_sampling_frequency_ignored():
    r = np.random.default_rng(3).standard_normal(50)
    rv1 = realized_variance(r, sampling_frequency="5min")
    rv2 = realized_variance(r, sampling_frequency="1min")
    assert rv1 == pytest.approx(rv2)


def test_series_input():
    import pandas as pd
    r = pd.Series(np.random.default_rng(4).standard_normal(50))
    rv = realized_variance(r)
    assert isinstance(rv, float)


def test_dataframe_input():
    import pandas as pd
    rng = np.random.default_rng(5)
    df = pd.DataFrame(rng.standard_normal((10, 50)))
    rv = realized_variance(df)
    assert len(rv) == 10


def test_2d_annualize():
    rng = np.random.default_rng(6)
    data = rng.standard_normal((5, 78))
    rv_d = realized_variance(data)
    rv_a = realized_variance(data, annualize=True, periods_per_year=252)
    np.testing.assert_allclose(rv_a, rv_d * 252)
