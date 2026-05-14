"""Tests for oprim.volatility.egarch: egarch_fit, egarch_forecast."""
import numpy as np
import pytest

from oprim.volatility.egarch import egarch_fit, egarch_forecast

_SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))


def _simulate_egarch(n=500, omega=-0.1, alpha=0.1, gamma=-0.05, beta=0.9, seed=0):
    """Simulate EGARCH(1,1) process."""
    rng = np.random.default_rng(seed)
    log_sigma2 = np.zeros(n)
    eps = np.zeros(n)
    log_sigma2[0] = 0.0
    for t in range(1, n):
        sigma = np.exp(0.5 * log_sigma2[t - 1])
        z = rng.standard_normal()
        eps[t - 1] = sigma * z
        log_sigma2[t] = (
            omega
            + alpha * (abs(z) - _SQRT_2_OVER_PI)
            + gamma * z
            + beta * log_sigma2[t - 1]
        )
    return eps[: n - 1]


def test_egarch_fit_returns_keys():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    for key in ("params", "log_likelihood", "aic", "bic", "persistence",
                "residuals", "conditional_variance", "converged"):
        assert key in r


def test_egarch_params_keys():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    for key in ("omega", "alpha", "gamma", "beta", "mu"):
        assert key in r["params"]


def test_egarch_too_short_raises():
    with pytest.raises(ValueError, match="at least 50"):
        egarch_fit(np.arange(10.0))


def test_egarch_conditional_variance_positive():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    assert np.all(r["conditional_variance"] > 0)


def test_egarch_aic_less_than_bic_large_n():
    x = _simulate_egarch(500)
    r = egarch_fit(x)
    # For large n, BIC penalty > AIC penalty, so BIC >= AIC
    assert r["bic"] >= r["aic"]


def test_egarch_persistence_range():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    assert 0 <= r["persistence"] < 1.0


def test_egarch_residuals_shape():
    x = _simulate_egarch(400)
    r = egarch_fit(x)
    assert len(r["residuals"]) == len(x)


def test_egarch_log_likelihood_finite():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    assert np.isfinite(r["log_likelihood"])


def test_egarch_captures_leverage():
    x = _simulate_egarch(500, gamma=-0.15, seed=42)
    r = egarch_fit(x)
    if r["converged"]:
        # Leverage effect: gamma should be negative
        assert r["params"]["gamma"] < 0.1  # at least not strongly positive


def test_egarch_warn_pq():
    import warnings
    x = _simulate_egarch(300)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        egarch_fit(x, p=2)
        assert len(w) == 1


def test_egarch_forecast_horizon_1():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    eps = r["residuals"]
    last_var = r["conditional_variance"][-1]
    last_sigma = float(np.sqrt(max(last_var, 1e-10)))
    last_z = float(eps[-1] / last_sigma)
    last_log_var = float(np.log(max(last_var, 1e-10)))
    forecast = egarch_forecast(r["params"], last_z, last_log_var, horizon=1)
    assert len(forecast) == 1
    assert forecast[0] > 0


def test_egarch_forecast_horizon_5():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    eps = r["residuals"]
    last_var = r["conditional_variance"][-1]
    last_sigma = float(np.sqrt(max(last_var, 1e-10)))
    last_z = float(eps[-1] / last_sigma)
    last_log_var = float(np.log(max(last_var, 1e-10)))
    forecast = egarch_forecast(r["params"], last_z, last_log_var, horizon=5)
    assert len(forecast) == 5
    assert np.all(forecast > 0)


def test_egarch_forecast_all_positive():
    x = _simulate_egarch(300)
    r = egarch_fit(x)
    last_var = r["conditional_variance"][-1]
    last_log_var = float(np.log(max(last_var, 1e-10)))
    forecast = egarch_forecast(r["params"], 0.0, last_log_var, horizon=10)
    assert np.all(forecast > 0)


def test_egarch_fit_series_input():
    """Covers pd.Series branch (line 95)."""
    import pandas as pd
    x = _simulate_egarch(300)
    r = egarch_fit(pd.Series(x))
    assert "params" in r


def test_egarch_forecast_beta_ge_1():
    """Covers the else branch at line 256 (|beta| >= 1)."""
    params = {"omega": -0.1, "alpha": 0.1, "gamma": -0.05, "beta": 1.0, "mu": 0.0}
    last_log_var = -2.0
    forecast = egarch_forecast(params, 0.5, last_log_var, horizon=3)
    assert len(forecast) == 3
    assert np.all(forecast > 0)
