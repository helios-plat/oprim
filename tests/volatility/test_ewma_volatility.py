"""Tests for oprim.volatility.ewma_volatility."""
import math
import numpy as np
import pandas as pd
import pytest

from oprim.volatility.ewma import ewma_volatility


class TestEWMAVolatility:
    def test_ewma_vol_default_lambda_94(self):
        """Default lambda_=0.94, result has same length as input."""
        rng = np.random.default_rng(0)
        r = rng.standard_normal(100) * 0.01
        result = ewma_volatility(r)
        assert len(result) == len(r)
        assert np.all(result > 0)

    def test_ewma_vol_constant_returns_converges(self):
        """Constant returns: eventually converge to known sigma."""
        # With r_t = c, the recursive formula eventually converges
        c = 0.01
        r = np.full(200, c)
        result = ewma_volatility(r, lambda_=0.94)
        # steady state: sigma^2 = (1-lambda) * c^2 / (1-lambda) = c^2
        # Actually: sigma^2 = lambda*sigma^2 + (1-lambda)*c^2
        # => sigma^2(1-lambda) = (1-lambda)*c^2 => sigma^2 = c^2
        assert result[-1] == pytest.approx(abs(c), rel=0.01)

    def test_ewma_vol_initial_variance_provided(self):
        """Provide initial_variance, check first output matches formula."""
        r = np.array([0.01, 0.02, 0.03])
        lambda_ = 0.94
        init_var = 0.0004
        result = ewma_volatility(r, lambda_=lambda_, initial_variance=init_var)
        # sigma2[0] = lambda_*init_var + (1-lambda_)*r[0]^2
        expected_var = lambda_ * init_var + (1.0 - lambda_) * r[0] ** 2
        assert result[0] == pytest.approx(math.sqrt(expected_var), rel=1e-12)

    def test_ewma_vol_initial_variance_auto_estimated(self):
        """Without initial_variance, seed uses var(returns)."""
        rng = np.random.default_rng(42)
        r = rng.standard_normal(50) * 0.02
        result_auto = ewma_volatility(r)
        result_manual = ewma_volatility(r, initial_variance=np.var(r))
        np.testing.assert_allclose(result_auto, result_manual, rtol=1e-12)

    def test_ewma_vol_annualized(self):
        """annualize=True multiplies by sqrt(252)."""
        r = np.random.default_rng(1).standard_normal(50) * 0.01
        result_raw = ewma_volatility(r, annualize=False)
        result_ann = ewma_volatility(r, annualize=True, periods_per_year=252)
        np.testing.assert_allclose(result_ann, result_raw * math.sqrt(252), rtol=1e-12)

    def test_ewma_vol_invalid_lambda_zero_raises(self):
        """lambda_=0 → ValueError."""
        with pytest.raises(ValueError, match="lambda_"):
            ewma_volatility([0.01, 0.02], lambda_=0.0)

    def test_ewma_vol_invalid_lambda_one_raises(self):
        """lambda_=1 → ValueError."""
        with pytest.raises(ValueError, match="lambda_"):
            ewma_volatility([0.01, 0.02], lambda_=1.0)

    def test_ewma_vol_pandas_series_preserves_index(self):
        """pd.Series input preserves index."""
        idx = pd.date_range("2023-01-01", periods=5)
        s = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01], index=idx)
        result = ewma_volatility(s)
        assert isinstance(result, pd.Series)
        assert list(result.index) == list(idx)

    def test_ewma_vol_high_vol_period_increases_estimate(self):
        """After high-vol shock, EWMA estimate should increase."""
        r = np.zeros(100)
        r[50] = 0.10  # big shock
        result = ewma_volatility(r, lambda_=0.94, initial_variance=0.0001)
        # sigma after shock should be bigger than before
        assert result[51] > result[49]

    @pytest.mark.academic_reference
    def test_ewma_vol_riskmetrics_formula(self):
        """RiskMetrics (1996): sigma_2^2 = 0.94*sigma_1^2 + 0.06*r_1^2, rtol=1e-12."""
        r1 = 0.01
        lambda_ = 0.94
        init_var = 0.0004  # sigma_0^2
        r = np.array([r1, 0.0])
        result = ewma_volatility(r, lambda_=lambda_, initial_variance=init_var)
        # sigma2[0] = 0.94 * init_var + 0.06 * r1^2
        sigma1_sq = lambda_ * init_var + (1.0 - lambda_) * r1**2
        # sigma2[1] = 0.94 * sigma1_sq + 0.06 * r[0]^2
        sigma2_sq = lambda_ * sigma1_sq + (1.0 - lambda_) * r1**2
        # result[1] corresponds to sigma2
        assert result[1] ** 2 == pytest.approx(sigma2_sq, rel=1e-12)
