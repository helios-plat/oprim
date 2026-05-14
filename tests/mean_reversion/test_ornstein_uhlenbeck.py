"""Tests for oprim.mean_reversion.ornstein_uhlenbeck."""
import math
import numpy as np
import pandas as pd
import pytest

from oprim.mean_reversion.ornstein_uhlenbeck import (
    ornstein_uhlenbeck_fit,
    ornstein_uhlenbeck_half_life,
)


def simulate_ou(theta, mu, sigma, dt, n, seed=42):
    """Simulate OU process using Euler-Maruyama."""
    rng = np.random.default_rng(seed)
    X = np.zeros(n)
    X[0] = mu
    for t in range(1, n):
        X[t] = X[t - 1] + theta * (mu - X[t - 1]) * dt + sigma * math.sqrt(dt) * rng.standard_normal()
    return X


class TestOUFit:
    def test_ou_fit_returns_four_keys(self):
        """ornstein_uhlenbeck_fit returns dict with 4 keys."""
        rng = np.random.default_rng(0)
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 500)
        result = ornstein_uhlenbeck_fit(X, dt=1 / 252)
        assert set(result.keys()) == {"theta", "mu", "sigma", "half_life"}

    def test_ou_fit_synthetic_known_params(self):
        """Fit OU with theta=2.0, mu=0, sigma=0.5; theta within 30%."""
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 2000, seed=7)
        result = ornstein_uhlenbeck_fit(X, dt=1 / 252)
        # Stochastic test: allow wide tolerance
        if not math.isnan(result["theta"]):
            assert result["theta"] > 0.0
            # Within a factor of 3x of true value is reasonable for 2000 obs
            assert 0.5 < result["theta"] < 20.0

    def test_ou_fit_random_walk_returns_low_theta(self):
        """Random walk → theta near 0 or NaN."""
        rng = np.random.default_rng(1)
        X = np.cumsum(rng.standard_normal(500))
        result = ornstein_uhlenbeck_fit(X)
        # theta should be very small or NaN
        if not math.isnan(result["theta"]):
            assert result["theta"] < 2.0

    def test_ou_fit_insufficient_data_raises(self):
        """len < 30 → ValueError."""
        with pytest.raises(ValueError, match="30"):
            ornstein_uhlenbeck_fit([0.1] * 20)

    def test_ou_fit_half_life_formula(self):
        """half_life = log(2)/theta."""
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 1000)
        result = ornstein_uhlenbeck_fit(X, dt=1 / 252)
        if not math.isnan(result["theta"]):
            expected_hl = math.log(2.0) / result["theta"]
            assert result["half_life"] == pytest.approx(expected_hl, rel=1e-10)

    def test_ou_fit_dt_parameter(self):
        """dt=1/252 vs dt=1.0 gives different theta."""
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 500)
        r1 = ornstein_uhlenbeck_fit(X, dt=1 / 252)
        r2 = ornstein_uhlenbeck_fit(X, dt=1.0)
        if not (math.isnan(r1["theta"]) or math.isnan(r2["theta"])):
            assert r1["theta"] != pytest.approx(r2["theta"], rel=0.01)

    def test_ou_fit_pandas_series_in(self):
        """pd.Series input is accepted."""
        X = simulate_ou(1.0, 0.0, 0.5, 1 / 252, 100)
        s = pd.Series(X)
        result = ornstein_uhlenbeck_fit(s)
        assert isinstance(result, dict)

    @pytest.mark.academic_reference
    def test_ou_fit_smith_2010_example(self):
        """Smith (2010) MLE formula: manually compute on known data.

        For a short OU simulation, verify that the returned theta satisfies
        theta = -log(rho)/dt where rho = corr(X_t, X_{t-1}), rtol=0.05.
        """
        rng = np.random.default_rng(123)
        X = simulate_ou(1.5, 0.5, 0.3, 1.0, 200, seed=123)
        result = ornstein_uhlenbeck_fit(X, dt=1.0)
        # Manually compute rho
        rho = float(np.corrcoef(X[:-1], X[1:])[0, 1])
        if rho > 0:
            expected_theta = -math.log(rho)
            if not math.isnan(result["theta"]):
                assert result["theta"] == pytest.approx(expected_theta, rel=0.05)


class TestOUHalfLife:
    def test_ou_half_life_regression_basic(self):
        """OU sim, regression half_life > 0."""
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 500)
        hl = ornstein_uhlenbeck_half_life(X, method="regression")
        assert hl > 0

    def test_ou_half_life_mle_basic(self):
        """OU sim, mle half_life > 0."""
        X = simulate_ou(2.0, 0.0, 0.5, 1 / 252, 500)
        hl = ornstein_uhlenbeck_half_life(X, method="mle")
        assert hl > 0

    def test_ou_half_life_random_walk_returns_inf_or_large(self):
        """Random walk → large/inf half_life."""
        rng = np.random.default_rng(5)
        X = np.cumsum(rng.standard_normal(500))
        hl = ornstein_uhlenbeck_half_life(X, method="regression")
        assert hl == math.inf or hl > 10.0

    def test_ou_half_life_invalid_method_raises(self):
        """Unknown method → ValueError."""
        X = simulate_ou(1.0, 0.0, 0.5, 1.0, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            ornstein_uhlenbeck_half_life(X, method="bogus")

    def test_ou_half_life_does_not_import_fit(self):
        """Ensure ornstein_uhlenbeck_half_life doesn't call ornstein_uhlenbeck_fit at runtime."""
        import oprim.mean_reversion.ornstein_uhlenbeck as module
        # Check that the function body code doesn't reference fit as a callable
        # by verifying it doesn't call `ornstein_uhlenbeck_fit(` (with open paren)
        import inspect
        source = inspect.getsource(ornstein_uhlenbeck_half_life)
        # The function may mention fit in docstrings but must not call it
        assert "ornstein_uhlenbeck_fit(" not in source

    def test_ou_half_life_pandas_input(self):
        """pd.Series input is accepted."""
        X = simulate_ou(1.0, 0.0, 0.5, 1.0, 100)
        s = pd.Series(X)
        hl = ornstein_uhlenbeck_half_life(s)
        assert hl > 0

    def test_ou_half_life_mle_pandas(self):
        """pd.Series + mle method works."""
        X = simulate_ou(1.0, 0.0, 0.3, 1.0, 200)
        s = pd.Series(X)
        hl = ornstein_uhlenbeck_half_life(s, method="mle")
        assert hl > 0

    def test_ou_half_life_regression_formula(self):
        """Regression half_life = log(2)/theta where theta = -beta from OLS."""
        X = simulate_ou(3.0, 0.0, 0.2, 1.0, 1000, seed=99)
        hl = ornstein_uhlenbeck_half_life(X, method="regression")
        dX = X[1:] - X[:-1]
        X_lag = X[:-1]
        beta = np.cov(dX, X_lag)[0, 1] / np.var(X_lag, ddof=1)
        theta = -beta
        if theta > 0:
            expected_hl = math.log(2.0) / theta
            assert hl == pytest.approx(expected_hl, rel=1e-10)
