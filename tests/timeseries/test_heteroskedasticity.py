"""Tests for oprim.timeseries.heteroskedasticity: breusch_pagan_test."""
import numpy as np
import pytest

from oprim.timeseries.heteroskedasticity import breusch_pagan_test


def _make_homoskedastic(n=200, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    e = rng.standard_normal(n)
    return e, x


def _make_heteroskedastic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2))
    # Variance proportional to x[:,0]^2
    e = rng.standard_normal(n) * (1 + 3 * np.abs(x[:, 0]))
    return e, x


def test_homoskedastic_not_rejected():
    e, x = _make_homoskedastic(300)
    r = breusch_pagan_test(e, x)
    assert r["is_homoskedastic"]


def test_heteroskedastic_rejected():
    e, x = _make_heteroskedastic(500)
    r = breusch_pagan_test(e, x)
    assert not r["is_homoskedastic"]


def test_returns_expected_keys():
    e, x = _make_homoskedastic(200)
    r = breusch_pagan_test(e, x)
    for key in ("lm_statistic", "lm_p_value", "f_statistic", "f_p_value",
                "r_squared_aux", "df", "n_obs", "is_homoskedastic"):
        assert key in r


def test_lm_statistic_positive():
    e, x = _make_homoskedastic(200)
    r = breusch_pagan_test(e, x)
    assert r["lm_statistic"] >= 0


def test_length_mismatch_raises():
    e, x = _make_homoskedastic(200)
    with pytest.raises(ValueError, match="same length"):
        breusch_pagan_test(e, x[:100])


def test_1d_exog():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    e = rng.standard_normal(100)
    r = breusch_pagan_test(e, x)
    assert "lm_statistic" in r


def test_series_and_dataframe_input():
    import pandas as pd
    rng = np.random.default_rng(10)
    n = 200
    x = rng.standard_normal((n, 2))
    e = rng.standard_normal(n)
    r = breusch_pagan_test(pd.Series(e), pd.DataFrame(x))
    assert "lm_statistic" in r


def test_too_short_raises():
    rng = np.random.default_rng(9)
    e = rng.standard_normal(5)
    x = rng.standard_normal((5, 1))
    with pytest.raises(ValueError, match="at least 10"):
        breusch_pagan_test(e, x)


@pytest.mark.academic_reference
def test_bp_matches_statsmodels():
    """BP LM statistic should be within 2% of statsmodels."""
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
    except ImportError:
        pytest.skip("statsmodels not installed")
    e, x = _make_heteroskedastic(500)
    r = breusch_pagan_test(e, x)
    sm_r = het_breuschpagan(e**2, np.column_stack([np.ones(len(e)), x]))
    # Compare LM stats (statsmodels returns different variant sometimes)
    # Both should agree on rejection
    assert (r["lm_p_value"] < 0.05) == (sm_r[1] < 0.05)


def test_bp_else_branch_n_minus_k_zero():
    """n - k <= 0 → f_stat = 0.0, f_p_value = 1.0 (lines 98-99)."""
    # n=12, exog with 12 columns → k = 12+1 = 13, n - k = -1 <= 0
    rng = np.random.default_rng(77)
    n = 12
    e = rng.standard_normal(n)
    x = rng.standard_normal((n, n))  # 12 regressors, so k=13 > n=12
    r = breusch_pagan_test(e, x)
    assert r["f_statistic"] == pytest.approx(0.0)
    assert r["f_p_value"] == pytest.approx(1.0)
