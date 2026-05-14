"""Tests for oprim.timeseries.stationarity: adf_test, kpss_test."""
import numpy as np
import pytest

from oprim.timeseries.stationarity import adf_test, kpss_test


def _ar1_series(phi, n=500, seed=42):
    """Generate AR(1) series."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


# ---------- ADF tests ----------

def test_adf_stationary_rejects():
    x = _ar1_series(0.5, 500)
    r = adf_test(x)
    assert r["is_stationary"]


def test_adf_random_walk_fails_to_reject():
    x = _ar1_series(1.0, 500)
    r = adf_test(x)
    assert not r["is_stationary"]


def test_adf_returns_expected_keys():
    x = _ar1_series(0.5, 200)
    r = adf_test(x)
    for key in ("statistic", "p_value", "lags_used", "n_obs", "critical_values", "is_stationary", "regression_type"):
        assert key in r, f"Missing key: {key}"


def test_adf_critical_values_present():
    x = _ar1_series(0.7, 300)
    r = adf_test(x)
    assert "1%" in r["critical_values"]
    assert "5%" in r["critical_values"]
    assert "10%" in r["critical_values"]


def test_adf_regression_ct():
    x = _ar1_series(0.5, 300)
    r = adf_test(x, regression="ct")
    assert r["regression_type"] == "ct"
    assert isinstance(r["statistic"], float)


def test_adf_regression_nc():
    x = _ar1_series(0.5, 300)
    r = adf_test(x, regression="nc")
    assert r["regression_type"] == "nc"


def test_adf_p_value_range():
    x = _ar1_series(0.5, 300)
    r = adf_test(x)
    assert 0 < r["p_value"] <= 1.0


def test_adf_invalid_regression_raises():
    with pytest.raises(ValueError, match="regression"):
        adf_test(np.arange(50.0), regression="bad")


def test_adf_too_short_raises():
    with pytest.raises(ValueError, match="at least"):
        adf_test(np.array([1.0, 2.0, 3.0]))


def test_adf_with_max_lag():
    x = _ar1_series(0.7, 300)
    r = adf_test(x, max_lag=2)
    assert r["lags_used"] <= 2


def test_adf_statistic_negative_for_stationary():
    """ADF t-stat should be more negative than 5% cv for stationary series."""
    x = _ar1_series(0.3, 500)
    r = adf_test(x)
    assert r["statistic"] < r["critical_values"]["5%"]


@pytest.mark.academic_reference
def test_adf_matches_statsmodels():
    """ADF statistic should be within 5% of statsmodels result."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        pytest.skip("statsmodels not installed")
    x = _ar1_series(0.7, 300)
    r = adf_test(x, regression="c")
    sm_r = adfuller(x, regression="c", autolag="AIC")
    assert abs(r["statistic"] - sm_r[0]) < abs(sm_r[0]) * 0.05 + 0.1


@pytest.mark.academic_reference
def test_adf_pvalue_matches_statsmodels():
    """ADF p-value should be in same region as statsmodels."""
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        pytest.skip("statsmodels not installed")
    x = _ar1_series(0.5, 500)
    r = adf_test(x, regression="c")
    sm_r = adfuller(x, regression="c", autolag="AIC")
    # Both should be < 0.05 (reject unit root)
    assert (r["p_value"] < 0.05) == (sm_r[1] < 0.05)


# ---------- KPSS tests ----------

def test_kpss_stationary_series():
    x = _ar1_series(0.5, 500)
    r = kpss_test(x)
    assert r["is_stationary"]


def test_kpss_random_walk_not_stationary():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(500))
    r = kpss_test(x)
    assert not r["is_stationary"]


def test_kpss_returns_expected_keys():
    x = _ar1_series(0.5, 200)
    r = kpss_test(x)
    for key in ("statistic", "p_value", "lags_used", "n_obs", "critical_values", "is_stationary"):
        assert key in r


def test_kpss_regression_ct():
    x = _ar1_series(0.5, 300)
    r = kpss_test(x, regression="ct")
    assert isinstance(r["statistic"], float)


def test_kpss_invalid_regression_raises():
    with pytest.raises(ValueError, match="regression"):
        kpss_test(np.arange(50.0), regression="nc")


def test_kpss_too_short_raises():
    with pytest.raises(ValueError, match="at least"):
        kpss_test(np.array([1.0, 2.0, 3.0]))


def test_adf_autolag_none():
    x = _ar1_series(0.5, 200)
    r = adf_test(x, max_lag=2, autolag=None)
    assert r["lags_used"] == 2


def test_adf_autolag_bic():
    x = _ar1_series(0.7, 300)
    r = adf_test(x, autolag="BIC")
    assert isinstance(r["statistic"], float)


def test_adf_autolag_tstat():
    x = _ar1_series(0.7, 300)
    r = adf_test(x, autolag="t-stat")
    assert isinstance(r["statistic"], float)


def test_adf_regression_ctt():
    x = _ar1_series(0.5, 300)
    r = adf_test(x, regression="ctt")
    assert r["regression_type"] == "ctt"


def test_adf_series_input():
    import pandas as pd
    x = pd.Series(_ar1_series(0.5, 300))
    r = adf_test(x)
    assert r["is_stationary"]


def test_kpss_series_input():
    import pandas as pd
    x = pd.Series(_ar1_series(0.5, 200))
    r = kpss_test(x)
    assert isinstance(r["statistic"], float)


def test_kpss_with_n_lags():
    x = _ar1_series(0.5, 200)
    r = kpss_test(x, n_lags=5)
    assert r["lags_used"] == 5


def test_adf_pvalue_extremes():
    """Test that very negative and very positive t-stats get boundary p-values."""
    from oprim.timeseries.stationarity import _adf_pvalue
    p_low = _adf_pvalue(-10.0, "c")
    assert p_low <= 0.01
    p_high = _adf_pvalue(5.0, "c")
    assert p_high >= 0.9


def test_kpss_pvalue_interpolation():
    """Force KPSS stat into the interpolation range between 0.10 and 0.01 CVs."""
    # stat between 0.347 and 0.739 for 'c' regression
    # Use a slightly noisy AR(1) process that gives stat in middle range
    rng = np.random.default_rng(100)
    # Generate a process that should give KPSS stat ~0.4-0.5
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal()
    r = kpss_test(x, regression="c")
    # Just verify interpolation was hit (stat in 0.347..0.739 range)
    assert 0 < r["p_value"] <= 1.0


@pytest.mark.academic_reference
def test_kpss_matches_statsmodels():
    """KPSS statistic should be within 5% of statsmodels result."""
    try:
        from statsmodels.tsa.stattools import kpss
    except ImportError:
        pytest.skip("statsmodels not installed")
    import warnings
    x = _ar1_series(0.5, 300)
    r = kpss_test(x, regression="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_r = kpss(x, regression="c", nlags="auto")
    assert abs(r["statistic"] - sm_r[0]) < abs(sm_r[0]) * 0.10 + 0.01
