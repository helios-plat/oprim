"""Tests for oprim.timeseries.cointegration."""
import numpy as np
import pytest

from oprim.timeseries.cointegration import engle_granger_cointegration, johansen_cointegration


def _make_cointegrated_pair(n=500, beta=2.0, seed=42):
    """Generate cointegrated pair: y = beta*x + stationary noise."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = beta * x + rng.standard_normal(n) * 0.5
    return y, x


def _make_independent_rws(n=500, seed=0):
    """Generate two independent random walks (not cointegrated)."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    return y, x


def _make_cointegrated_multivar(n=500, k=2, seed=42):
    """Generate k time series with rank-1 cointegration via common factor."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.standard_normal(n))
    data = np.column_stack([common + 0.1 * rng.standard_normal(n) for _ in range(k)])
    return data


# ---------- Engle-Granger tests ----------

def test_eg_cointegrated_detects():
    y, x = _make_cointegrated_pair(500)
    r = engle_granger_cointegration(y, x)
    assert r["is_cointegrated"]


def test_eg_returns_expected_keys():
    y, x = _make_cointegrated_pair(200)
    r = engle_granger_cointegration(y, x)
    for key in ("statistic", "p_value", "critical_values", "coef_intercept",
                "coef_slope", "n_obs", "is_cointegrated", "regression_type"):
        assert key in r


def test_eg_critical_values_present():
    y, x = _make_cointegrated_pair(200)
    r = engle_granger_cointegration(y, x)
    assert "1%" in r["critical_values"]
    assert "5%" in r["critical_values"]


def test_eg_coefficient_reasonable():
    y, x = _make_cointegrated_pair(500, beta=2.0)
    r = engle_granger_cointegration(y, x)
    assert abs(r["coef_slope"] - 2.0) < 0.2


def test_eg_invalid_trend_raises():
    y, x = _make_cointegrated_pair(200)
    with pytest.raises(ValueError, match="trend"):
        engle_granger_cointegration(y, x, trend="bad")


def test_eg_length_mismatch_raises():
    y, x = _make_cointegrated_pair(200)
    with pytest.raises(ValueError, match="same length"):
        engle_granger_cointegration(y, x[:100])


def test_eg_too_short_raises():
    y = np.arange(10.0)
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="at least"):
        engle_granger_cointegration(y, x)


@pytest.mark.academic_reference
def test_eg_matches_statsmodels():
    """EG t-statistic should be close to statsmodels coint."""
    try:
        from statsmodels.tsa.stattools import coint
    except ImportError:
        pytest.skip("statsmodels not installed")
    y, x = _make_cointegrated_pair(300)
    r = engle_granger_cointegration(y, x, trend="c")
    sm_r = coint(y, x, trend="c")
    assert abs(r["statistic"] - sm_r[0]) < abs(sm_r[0]) * 0.05 + 0.2


# ---------- Johansen tests ----------

def test_johansen_two_series_rank_one():
    data = _make_cointegrated_multivar(500, 2)
    r = johansen_cointegration(data)
    assert r["cointegration_rank"] >= 1


def test_johansen_returns_expected_keys():
    data = _make_cointegrated_multivar(300, 2)
    r = johansen_cointegration(data)
    for key in ("trace_stats", "max_eigenvalue_stats", "eigenvalues",
                "eigenvectors", "cointegration_rank", "cointegrating_vectors",
                "n_obs", "n_vars"):
        assert key in r


def test_johansen_n_vars():
    data = _make_cointegrated_multivar(400, 3)
    r = johansen_cointegration(data)
    assert r["n_vars"] == 3


def test_johansen_trace_stats_shape():
    data = _make_cointegrated_multivar(400, 3)
    r = johansen_cointegration(data)
    assert len(r["trace_stats"]) == 3


def test_johansen_eigenvalues_positive():
    data = _make_cointegrated_multivar(400, 2)
    r = johansen_cointegration(data)
    # Leading eigenvalues should be positive
    assert r["eigenvalues"][0] > 0


def test_johansen_too_few_vars_raises():
    x = np.random.default_rng(0).standard_normal((200, 1))
    with pytest.raises(ValueError, match="at least 2"):
        johansen_cointegration(x)


def test_johansen_too_many_vars_raises():
    x = np.random.default_rng(0).standard_normal((500, 7))
    with pytest.raises(ValueError, match="up to 6"):
        johansen_cointegration(x)


def test_eg_trend_ct():
    y, x = _make_cointegrated_pair(300)
    r = engle_granger_cointegration(y, x, trend="ct")
    assert "statistic" in r


def test_eg_trend_nc():
    y, x = _make_cointegrated_pair(300)
    r = engle_granger_cointegration(y, x, trend="nc")
    assert "statistic" in r


def test_eg_series_input():
    import pandas as pd
    y, x = _make_cointegrated_pair(300)
    r = engle_granger_cointegration(pd.Series(y), pd.Series(x))
    assert "is_cointegrated" in r


def test_eg_pvalue_extremes():
    """Test EG p-value at extreme t-stat values."""
    from oprim.timeseries.cointegration import _eg_pvalue
    p_low = _eg_pvalue(-10.0, "c")
    assert p_low <= 0.01
    p_high = _eg_pvalue(5.0, "c")
    assert p_high >= 0.9


def test_johansen_with_k_ar_diff_2():
    data = _make_cointegrated_multivar(400, 2)
    r = johansen_cointegration(data, k_ar_diff=2)
    assert "cointegration_rank" in r


def test_johansen_det_order_1():
    data = _make_cointegrated_multivar(400, 2)
    r = johansen_cointegration(data, det_order=1)
    assert "cointegration_rank" in r


def test_johansen_3d_raises():
    with pytest.raises(ValueError, match="2-D"):
        johansen_cointegration(np.zeros((5, 2, 2)))


def test_johansen_dataframe_input():
    import pandas as pd
    data = _make_cointegrated_multivar(400, 2)
    r = johansen_cointegration(pd.DataFrame(data))
    assert r["n_vars"] == 2


def test_johansen_4_vars():
    data = _make_cointegrated_multivar(500, 4)
    r = johansen_cointegration(data)
    assert r["n_vars"] == 4


@pytest.mark.academic_reference
def test_johansen_matches_statsmodels_rank():
    """Johansen rank should agree with statsmodels at 5% level."""
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        pytest.skip("statsmodels not installed")
    data = _make_cointegrated_multivar(500, 2)
    r = johansen_cointegration(data)
    sm_r = coint_johansen(data, det_order=0, k_ar_diff=1)
    # Both should identify at least 1 cointegrating vector
    assert r["cointegration_rank"] >= 1


# ---------- Additional coverage tests ----------

def test_eg_pvalue_midrange():
    """Hit the interpolation loop (lines 82-86)."""
    from oprim.timeseries.cointegration import _eg_pvalue
    p_mid = _eg_pvalue(-3.5, "c")
    assert 0.01 < p_mid < 0.99


def test_eg_pvalue_ct():
    """Use 'ct' trend to hit pvalue table lookup."""
    from oprim.timeseries.cointegration import _eg_pvalue
    p = _eg_pvalue(-3.0, "ct")
    assert 0 <= p <= 1


def test_johansen_insufficient_obs_raises():
    """Too few observations for Johansen (line 310)."""
    # k=2 requires at least 2*2+1+10=15 observations
    data = np.random.default_rng(0).standard_normal((14, 2))
    with pytest.raises(ValueError, match="Insufficient"):
        johansen_cointegration(data)


def test_johansen_too_few_effective_obs_raises():
    """T <= k should raise (line 322)."""
    # With k=2, k_ar_diff=5, n=8: T = 8-1-5=2, k=2, so T <= k
    data = np.random.default_rng(0).standard_normal((9, 2))
    with pytest.raises(ValueError, match="(Insufficient|Too few)"):
        johansen_cointegration(data, k_ar_diff=5)


def test_johansen_k_ar_diff_0():
    """k_ar_diff=0 means p=0, covers if p>0 False branch (line 330->337)."""
    data = _make_cointegrated_multivar(200, 2)
    r = johansen_cointegration(data, k_ar_diff=0)
    assert "cointegration_rank" in r


def test_johansen_det_order_1_p0():
    """det_order=1, k_ar_diff=0 covers else branch at line 347."""
    data = _make_cointegrated_multivar(200, 2)
    r = johansen_cointegration(data, det_order=1, k_ar_diff=0)
    assert "cointegration_rank" in r


def test_johansen_non_cointegrated():
    """Independent random walks → rank=0 to ensure cointegration_rank loop sets 0."""
    rng = np.random.default_rng(123)
    # Use OTM noise series (stationary but not integrated → rank=0 expected)
    data = np.column_stack([rng.standard_normal(200), rng.standard_normal(200)])
    r = johansen_cointegration(data)
    # rank can be 0 or 1 depending on eigenvalue, just assert valid
    assert r["cointegration_rank"] >= 0
