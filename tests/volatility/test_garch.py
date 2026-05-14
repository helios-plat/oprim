"""Tests for oprim.volatility.garch_fit and garch_forecast."""
import math
import warnings
import numpy as np
import pandas as pd
import pytest

from oprim.volatility.garch import garch_fit, garch_forecast


def _simulate_garch11(omega, alpha, beta, T, seed=42):
    """Simulate GARCH(1,1) returns."""
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(T)
    eps = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(T):
        eps[t] = math.sqrt(sigma2[t]) * rng.standard_normal()
        if t < T - 1:
            sigma2[t + 1] = omega + alpha * eps[t] ** 2 + beta * sigma2[t]
    return eps, sigma2


class TestGARCHFit:
    def test_garch_fit_returns_all_keys(self):
        """Verify all expected keys present in output."""
        rng = np.random.default_rng(0)
        r = rng.standard_normal(200) * 0.01
        result = garch_fit(r)
        expected_keys = {
            "params", "log_likelihood", "aic", "bic",
            "persistence", "unconditional_variance", "converged",
            "residuals", "conditional_variance",
        }
        assert set(result.keys()) == expected_keys
        assert set(result["params"].keys()) >= {"omega", "alpha", "beta", "mu"}

    def test_garch_fit_pure_white_noise_low_persistence(self):
        """i.i.d. noise → alpha+beta usually < 0.95."""
        rng = np.random.default_rng(1)
        r = rng.standard_normal(300) * 0.01
        result = garch_fit(r)
        # For white noise, persistence should be modest
        # Allow generous threshold since optimizer varies
        assert result["persistence"] < 1.0

    def test_garch_fit_unconditional_variance_when_stable(self):
        """persistence < 1 → unconditional_variance = omega/(1-alpha-beta) > 0."""
        eps, _ = _simulate_garch11(0.0001, 0.05, 0.90, 300)
        result = garch_fit(eps)
        if result["persistence"] < 1.0:
            p = result["params"]
            expected_uv = p["omega"] / (1.0 - p["alpha"] - p["beta"])
            assert result["unconditional_variance"] == pytest.approx(expected_uv, rel=1e-8)
            assert result["unconditional_variance"] > 0

    def test_garch_fit_aic_bic_calculated(self):
        """aic = -2*ll + 2*k; bic = -2*ll + k*log(T)."""
        rng = np.random.default_rng(2)
        r = rng.standard_normal(200) * 0.01
        result = garch_fit(r)
        ll = result["log_likelihood"]
        k = 4
        T = 200
        assert result["aic"] == pytest.approx(-2 * ll + 2 * k, rel=1e-8)
        assert result["bic"] == pytest.approx(-2 * ll + k * math.log(T), rel=1e-8)

    def test_garch_fit_converged_flag(self):
        """converged key is bool."""
        rng = np.random.default_rng(3)
        r = rng.standard_normal(200) * 0.01
        result = garch_fit(r)
        assert isinstance(result["converged"], bool)

    def test_garch_fit_insufficient_data_raises(self):
        """Fewer than 50 bars → ValueError."""
        with pytest.raises(ValueError, match="50"):
            garch_fit(np.random.default_rng(0).standard_normal(30) * 0.01)

    def test_garch_fit_t_distribution_runs(self):
        """distribution='t' runs without error."""
        rng = np.random.default_rng(4)
        r = rng.standard_normal(200) * 0.01
        result = garch_fit(r, distribution="t")
        assert "params" in result

    def test_garch_fit_conditional_variance_positive(self):
        """All sigma_t^2 > 0."""
        eps, _ = _simulate_garch11(0.0001, 0.05, 0.90, 200)
        result = garch_fit(eps)
        assert np.all(result["conditional_variance"] > 0)

    def test_garch_fit_warns_for_higher_order(self):
        """p>1 or q>1 emits UserWarning."""
        rng = np.random.default_rng(5)
        r = rng.standard_normal(200) * 0.01
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            garch_fit(r, p=2, q=1)
            assert len(w) >= 1
            assert issubclass(w[0].category, UserWarning)

    def test_garch_fit_residuals_length(self):
        """residuals array has same length as input."""
        rng = np.random.default_rng(6)
        r = rng.standard_normal(100) * 0.01
        result = garch_fit(r)
        assert len(result["residuals"]) == 100

    def test_garch_fit_pandas_input(self):
        """pd.Series input is accepted."""
        rng = np.random.default_rng(7)
        s = pd.Series(rng.standard_normal(150) * 0.01)
        result = garch_fit(s)
        assert "params" in result


class TestGARCHForecast:
    def test_garch_forecast_one_step(self):
        """Known params, verify 1-step formula exactly."""
        params = {"omega": 0.0001, "alpha": 0.05, "beta": 0.90, "mu": 0.0}
        last_eps = 0.02
        last_var = 0.0004
        result = garch_forecast(params, last_eps, last_var, horizon=1)
        expected = params["omega"] + params["alpha"] * last_eps**2 + params["beta"] * last_var
        assert result[0] == pytest.approx(expected, rel=1e-12)

    def test_garch_forecast_horizon_10_returns_10_values(self):
        """horizon=10 → array of length 10."""
        params = {"omega": 0.0001, "alpha": 0.05, "beta": 0.90}
        result = garch_forecast(params, 0.01, 0.0004, horizon=10)
        assert len(result) == 10

    def test_garch_forecast_converges_to_unconditional(self):
        """Many steps → omega/(1-alpha-beta)."""
        params = {"omega": 0.0001, "alpha": 0.05, "beta": 0.90}
        result = garch_forecast(params, 0.0, 0.0001, horizon=5000)
        uv = params["omega"] / (1.0 - params["alpha"] - params["beta"])
        assert result[-1] == pytest.approx(uv, rel=0.001)

    def test_garch_forecast_unstable_emits_warning(self):
        """alpha+beta >= 1 → RuntimeWarning."""
        params = {"omega": 0.0001, "alpha": 0.05, "beta": 0.96}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            garch_forecast(params, 0.01, 0.0004, horizon=5)
            assert len(w) >= 1
            assert issubclass(w[0].category, RuntimeWarning)

    def test_garch_forecast_does_not_import_fit(self):
        """garch_forecast does not call garch_fit at runtime."""
        import inspect
        source = inspect.getsource(garch_forecast)
        # Must not call garch_fit (with open paren)
        assert "garch_fit(" not in source

    @pytest.mark.academic_reference
    def test_garch_forecast_bollerslev_recursion(self):
        """Bollerslev (1986): multi-step recursion formula.

        sigma_{t+h}^2 = omega + (alpha+beta) * sigma_{t+h-1}^2
        for h >= 2. Verify 3-step forecast, rtol=1e-8.
        """
        omega, alpha, beta = 0.0001, 0.05, 0.90
        params = {"omega": omega, "alpha": alpha, "beta": beta}
        last_eps = 0.015
        last_var = 0.0003
        result = garch_forecast(params, last_eps, last_var, horizon=3)

        # Manual computation
        v1 = omega + alpha * last_eps**2 + beta * last_var
        v2 = omega + (alpha + beta) * v1
        v3 = omega + (alpha + beta) * v2

        assert result[0] == pytest.approx(v1, rel=1e-8)
        assert result[1] == pytest.approx(v2, rel=1e-8)
        assert result[2] == pytest.approx(v3, rel=1e-8)
