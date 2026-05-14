"""Tests for oprim.timeseries.distribution_tests: jarque_bera_test."""
import numpy as np
import pytest

from oprim.timeseries.distribution_tests import jarque_bera_test


def test_jb_normal_data_high_pvalue():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    r = jarque_bera_test(x)
    assert r["is_normal"]
    assert r["p_value"] > 0.05


def test_jb_skewed_data_rejects():
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=1.0, size=1000)
    r = jarque_bera_test(x)
    assert not r["is_normal"]
    assert r["p_value"] < 0.05


def test_jb_returns_expected_keys():
    x = np.random.default_rng(1).standard_normal(200)
    r = jarque_bera_test(x)
    for key in ("statistic", "p_value", "skewness", "kurtosis", "excess_kurtosis", "is_normal"):
        assert key in r


def test_jb_normal_skewness_near_zero():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000)
    r = jarque_bera_test(x)
    assert abs(r["skewness"]) < 0.2


def test_jb_normal_excess_kurtosis_near_zero():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(5000)
    r = jarque_bera_test(x)
    assert abs(r["excess_kurtosis"]) < 0.3


def test_jb_too_short_raises():
    with pytest.raises(ValueError, match="at least 8"):
        jarque_bera_test(np.array([1.0, 2.0, 3.0]))


def test_jb_constant_series():
    """Constant series returns p=1.0, is_normal=True."""
    x = np.ones(50)
    r = jarque_bera_test(x)
    assert r["is_normal"]


def test_jb_series_input():
    import pandas as pd
    x = pd.Series(np.random.default_rng(3).standard_normal(200))
    r = jarque_bera_test(x)
    assert "statistic" in r


@pytest.mark.academic_reference
def test_jb_matches_scipy():
    """JB statistic should match scipy.stats.jarque_bera."""
    from scipy.stats import jarque_bera
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    r = jarque_bera_test(x)
    sp_stat, sp_p = jarque_bera(x)
    assert abs(r["statistic"] - sp_stat) < abs(sp_stat) * 0.01 + 0.01
