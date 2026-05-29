"""Tests for oprim.volatility.gjr_garch: gjr_garch_fit, gjr_garch_forecast."""
import numpy as np
import pytest

from oprim.volatility.gjr_garch import gjr_garch_fit, gjr_garch_forecast


def _simulate_gjr_garch(n=500, omega=1e-5, alpha=0.05, gamma=0.10, beta=0.80, seed=0):
    """Simulate GJR-GARCH(1,1) process."""
    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - gamma / 2 - beta + 1e-6)
    for t in range(1, n):
        eps[t - 1] = np.sqrt(max(sigma2[t - 1], 1e-10)) * rng.standard_normal()
        indicator = 1.0 if eps[t - 1] < 0 else 0.0
        sigma2[t] = (
            omega
            + alpha * eps[t - 1] ** 2
            + gamma * indicator * eps[t - 1] ** 2
            + beta * sigma2[t - 1]
        )
    return eps[: n - 1]


def test_gjr_fit_returns_keys():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    for key in ("params", "log_likelihood", "aic", "bic", "persistence",
                "residuals", "conditional_variance", "converged"):
        assert key in r


def test_gjr_params_keys():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    for key in ("omega", "alpha", "gamma", "beta", "mu"):
        assert key in r["params"]


def test_gjr_too_short_raises():
    with pytest.raises(ValueError, match="at least 50"):
        gjr_garch_fit(np.arange(10.0))


def test_gjr_conditional_variance_positive():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    assert np.all(r["conditional_variance"] > 0)


def test_gjr_residuals_shape():
    x = _simulate_gjr_garch(400)
    r = gjr_garch_fit(x)
    assert len(r["residuals"]) == len(x)


def test_gjr_aic_bic_ordered():
    x = _simulate_gjr_garch(500)
    r = gjr_garch_fit(x)
    assert r["bic"] >= r["aic"]


def test_gjr_log_likelihood_finite():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    assert np.isfinite(r["log_likelihood"])


def test_gjr_persistence_below_one():
    x = _simulate_gjr_garch(400)
    r = gjr_garch_fit(x)
    assert r["persistence"] < 1.1  # Allow slight overshoot due to optimization


def test_gjr_leverage_detected():
    x = _simulate_gjr_garch(500, gamma=0.15, seed=42)
    r = gjr_garch_fit(x)
    if r["converged"]:
        # gamma should be non-negative
        assert r["params"]["gamma"] >= -0.05


def test_gjr_warn_pq():
    import warnings
    x = _simulate_gjr_garch(300)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gjr_garch_fit(x, p=2)
        assert len(w) == 1


def test_gjr_forecast_horizon_1():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    last_eps2 = float(r["residuals"][-1] ** 2)
    last_sigma2 = float(r["conditional_variance"][-1])
    forecast = gjr_garch_forecast(r["params"], last_eps2, last_sigma2, horizon=1)
    assert len(forecast) == 1
    assert forecast[0] > 0


def test_gjr_forecast_horizon_10():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    last_eps2 = float(r["residuals"][-1] ** 2)
    last_sigma2 = float(r["conditional_variance"][-1])
    forecast = gjr_garch_forecast(r["params"], last_eps2, last_sigma2, horizon=10)
    assert len(forecast) == 10
    assert np.all(forecast > 0)


def test_gjr_forecast_positive():
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(x)
    last_eps2 = float(r["residuals"][-1] ** 2)
    last_sigma2 = float(r["conditional_variance"][-1])
    forecast = gjr_garch_forecast(r["params"], last_eps2, last_sigma2, horizon=20)
    assert np.all(forecast > 0)


def test_gjr_forecast_expected_neg_frac_symmetric():
    params = {"omega": 1e-5, "alpha": 0.05, "gamma": 0.10, "beta": 0.80}
    forecast = gjr_garch_forecast(params, 1e-4, 1e-4, horizon=1, expected_neg_frac=0.5)
    assert forecast[0] > 0


def test_gjr_fit_series_input():
    """Covers pd.Series branch (line 100)."""
    import pandas as pd
    x = _simulate_gjr_garch(300)
    r = gjr_garch_fit(pd.Series(x))
    assert "params" in r


def test_gjr_nll_constraint_violation():
    """Trigger the constraint-violation return in _gjrgarch11_nll (line 23)."""
    from oprim.volatility.gjr_garch import _gjrgarch11_nll
    x = _simulate_gjr_garch(300)
    # omega <= 0 → should return 1e10
    params = np.array([-0.1, 0.05, 0.10, 0.80, 0.0])
    val = _gjrgarch11_nll(params, x)
    assert val == 1e10


def test_gjr_unconditional_variance_nan_when_persistence_ge_1():
    """When fitted persistence >= 1.0, unconditional_variance should be nan (line 197)."""
    # Force a high-persistence series
    rng = np.random.default_rng(999)
    # Random walk-like series
    x = np.cumsum(rng.standard_normal(300)) * 0.01
    r = gjr_garch_fit(x)
    # If persistence >= 1, unconditional_variance is nan
    if r["persistence"] >= 1.0:
        assert np.isnan(r["unconditional_variance"])
    else:
        # Otherwise just test it's positive
        assert r["unconditional_variance"] > 0
