"""Tests for oprim.timeseries.causality: granger_causality_test."""
import numpy as np
import pytest

from oprim.timeseries.causality import granger_causality_test


def _make_granger_causal(n=300, seed=42):
    """x -> y causality: y_t = 0.5 * x_{t-1} + noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * x[t - 1] + 0.3 * rng.standard_normal()
    return y, x


def _make_independent(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    return y, x


def test_causal_pair_detected():
    y, x = _make_granger_causal(300)
    r = granger_causality_test(y, x, max_lag=2)
    assert r["granger_causes"]


def test_independent_not_causal():
    y, x = _make_independent(300)
    r = granger_causality_test(y, x, max_lag=2)
    # Mostly should not reject for independent white noise
    # p-value typically > 0.05
    assert r["p_value"] > 0.01  # lenient: just check it's not wildly wrong


def test_returns_expected_keys():
    y, x = _make_independent(300)
    r = granger_causality_test(y, x)
    for key in ("f_statistic", "p_value", "max_lag", "n_obs",
                "ssr_restricted", "ssr_unrestricted", "granger_causes"):
        assert key in r


def test_f_statistic_positive():
    y, x = _make_granger_causal(300)
    r = granger_causality_test(y, x)
    assert r["f_statistic"] >= 0


def test_p_value_range():
    y, x = _make_independent(300)
    r = granger_causality_test(y, x)
    assert 0 <= r["p_value"] <= 1


def test_invalid_lag_raises():
    y, x = _make_independent(300)
    with pytest.raises(ValueError, match="max_lag"):
        granger_causality_test(y, x, max_lag=0)


def test_length_mismatch_raises():
    y, x = _make_independent(300)
    with pytest.raises(ValueError, match="same length"):
        granger_causality_test(y, x[:200])


def test_chi2_variant():
    y, x = _make_granger_causal(300)
    r = granger_causality_test(y, x, test="chi2")
    assert "chi2_statistic" in r


def test_series_input():
    import pandas as pd
    y, x = _make_granger_causal(300)
    r = granger_causality_test(pd.Series(y), pd.Series(x))
    assert "f_statistic" in r


def test_too_short_raises():
    y, x = _make_independent(15)  # need at least 2*max_lag+10=18 for max_lag=4
    with pytest.raises(ValueError, match="at least"):
        granger_causality_test(y, x, max_lag=4)


def test_too_few_dof_raises():
    """Very large max_lag relative to sample → df_denom <= 0 (line 110)."""
    rng = np.random.default_rng(7)
    n = 30  # exactly meets 2*max_lag+10=30 floor but df_denom = 20-21 = -1 <= 0
    y = rng.standard_normal(n)
    x = rng.standard_normal(n)
    # max_lag=10: n_eff=30-10=20, df_denom=20-21=-1 <= 0 → "degrees of freedom"
    with pytest.raises(ValueError, match="degrees of freedom"):
        granger_causality_test(y, x, max_lag=10)


def test_ssr_zero_gives_f_zero():
    """Constant y after differencing makes ssr_u=0 → f_stat=0 (line 112)."""
    # y is all the same constant; unrestricted residuals will be exactly 0
    n = 50
    y = np.zeros(n)
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n)
    r = granger_causality_test(y, x, max_lag=2)
    # f_stat should be 0 since ssr_u=0
    assert r["f_statistic"] == pytest.approx(0.0)
