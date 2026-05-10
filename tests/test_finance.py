"""Tests for oprim.finance module."""

import numpy as np
import pandas as pd
import pytest

from oprim.finance import beta_alpha_ols, drawdown_curve, sharpe_ratio, value_at_risk


# ============================================================
# drawdown_curve
# ============================================================
class TestDrawdownCurve:
    def test_no_drawdown(self):
        equity = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = drawdown_curve(equity)
        assert result["max_drawdown"] == 0.0

    def test_single_peak_valley(self):
        equity = pd.Series([100, 110, 90, 95, 105.0])
        result = drawdown_curve(equity)
        # Max drawdown from 110 to 90 = -20/110
        assert result["max_drawdown"] == pytest.approx(-20 / 110, abs=1e-9)

    def test_multi_peak(self):
        equity = pd.Series([100, 120, 80, 130, 60.0])
        result = drawdown_curve(equity)
        # Max drawdown from 130 to 60 = -70/130
        assert result["max_drawdown"] == pytest.approx(-70 / 130, abs=1e-9)

    def test_no_recovery(self):
        equity = pd.Series([100, 110, 90, 85, 80.0])
        result = drawdown_curve(equity)
        assert result["max_drawdown_recovery"] is None

    def test_from_returns(self):
        returns = pd.Series([0.1, -0.2, 0.05, 0.1])
        result = drawdown_curve(returns, input_type="returns")
        assert result["max_drawdown"] < 0

    def test_from_returns_non_compound(self):
        returns = pd.Series([0.1, -0.2, 0.05])
        result = drawdown_curve(returns, input_type="returns", compound=False)
        assert result["max_drawdown"] < 0

    def test_with_datetime_index(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        equity = pd.Series([100, 110, 90, 95, 115.0], index=idx)
        result = drawdown_curve(equity)
        assert isinstance(result["max_drawdown_start"], pd.Timestamp)
        assert result["underwater_duration_days"] >= 0


# ============================================================
# sharpe_ratio
# ============================================================
class TestSharpeRatio:
    def test_known_sharpe(self):
        """Constant positive returns → high Sharpe."""
        returns = pd.Series([0.01] * 252)
        sr = sharpe_ratio(returns)
        # mean=0.01, std≈0, but with ddof=1 std is 0 → NaN
        # Use slight variation
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.01, 252))
        sr = sharpe_ratio(returns)
        assert np.isfinite(sr)

    def test_zero_std(self):
        returns = pd.Series([0.01] * 100)
        assert np.isnan(sharpe_ratio(returns))

    def test_annualization(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 365))
        sr_252 = sharpe_ratio(returns, annualization_factor=252)
        sr_365 = sharpe_ratio(returns, annualization_factor=365)
        # Higher annualization → higher absolute Sharpe
        assert abs(sr_365) > abs(sr_252)

    def test_risk_free_series(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 100))
        rf = pd.Series(np.full(100, 0.0001))
        sr = sharpe_ratio(returns, risk_free_rate=rf)
        assert np.isfinite(sr)

    def test_academic_manual(self):
        """Manual Sharpe calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.005])
        excess = returns - 0.0
        expected = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252))
        result = sharpe_ratio(returns)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# beta_alpha_ols
# ============================================================
class TestBetaAlphaOLS:
    def test_single_factor(self):
        rng = np.random.default_rng(42)
        market = pd.Series(rng.normal(0.001, 0.02, 100))
        asset = 0.001 + 1.2 * market + pd.Series(rng.normal(0, 0.005, 100))
        result = beta_alpha_ols(asset, market, min_samples=10)
        assert abs(result["beta"] - 1.2) < 0.2
        assert result["r_squared"] > 0.5

    def test_multi_factor(self):
        rng = np.random.default_rng(42)
        factors = pd.DataFrame({
            "mkt": rng.normal(0.001, 0.02, 100),
            "smb": rng.normal(0, 0.01, 100),
        })
        asset = pd.Series(
            0.0005 + 1.0 * factors["mkt"] + 0.5 * factors["smb"] + rng.normal(0, 0.005, 100)
        )
        result = beta_alpha_ols(asset, factors, min_samples=10)
        assert isinstance(result["beta"], dict)
        assert "mkt" in result["beta"]

    def test_hac(self):
        rng = np.random.default_rng(42)
        market = pd.Series(rng.normal(0, 0.02, 100))
        asset = pd.Series(0.5 * market.values + rng.normal(0, 0.01, 100))
        result = beta_alpha_ols(asset, market, use_hac=True, min_samples=10)
        assert "alpha_se" in result

    def test_min_samples_raises(self):
        with pytest.raises(ValueError, match="samples"):
            beta_alpha_ols(pd.Series([0.01] * 10), pd.Series([0.02] * 10), min_samples=30)

    def test_academic_vs_statsmodels(self):
        """Compare with direct statsmodels OLS."""
        import statsmodels.api as sm

        rng = np.random.default_rng(42)
        market = rng.normal(0, 0.02, 200)
        asset = 0.001 + 0.8 * market + rng.normal(0, 0.005, 200)

        result = beta_alpha_ols(pd.Series(asset), pd.Series(market), min_samples=10)

        X = sm.add_constant(market)
        model = sm.OLS(asset, X).fit()
        np.testing.assert_allclose(result["alpha"], model.params[0], rtol=1e-9)
        np.testing.assert_allclose(result["beta"], model.params[1], rtol=1e-9)


# ============================================================
# value_at_risk
# ============================================================
class TestValueAtRisk:
    def test_historical(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        result = value_at_risk(returns, method="historical")
        assert result["var"] > 0
        assert result["es"] >= result["var"]

    def test_parametric(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        result = value_at_risk(returns, method="parametric")
        assert result["var"] > 0

    def test_cornish_fisher(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        result = value_at_risk(returns, method="cornish_fisher")
        assert result["var"] > 0

    def test_normal_methods_agree(self):
        """For normal distribution, all methods should give similar VaR."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 10000))
        hist = value_at_risk(returns, method="historical")
        param = value_at_risk(returns, method="parametric")
        cf = value_at_risk(returns, method="cornish_fisher")
        # Should be within 10% of each other
        np.testing.assert_allclose(hist["var"], param["var"], rtol=0.1)
        np.testing.assert_allclose(hist["var"], cf["var"], rtol=0.1)

    def test_es_gt_var(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(-0.001, 0.03, 1000))
        result = value_at_risk(returns, method="historical")
        assert result["es"] >= result["var"]

    def test_no_es(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 100))
        result = value_at_risk(returns, include_es=False)
        assert result["es"] is None

    def test_confidence_levels(self):
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        var_95 = value_at_risk(returns, confidence_level=0.95)
        var_99 = value_at_risk(returns, confidence_level=0.99)
        assert var_99["var"] > var_95["var"]


# ============================================================
# Additional coverage tests
# ============================================================
class TestFinanceExtra:
    def test_drawdown_recovery_with_timestamps(self):
        """Test recovery detection with datetime index."""
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        equity = pd.Series([100, 110, 90, 95, 100, 110, 115, 120, 125, 130.0], index=idx)
        result = drawdown_curve(equity)
        assert result["max_drawdown_recovery"] is not None
        assert result["underwater_duration_days"] > 0

    def test_drawdown_integer_index(self):
        """Test with integer index (non-timestamp)."""
        equity = pd.Series([100, 110, 80, 90, 120.0])
        result = drawdown_curve(equity)
        assert result["max_drawdown"] < 0
        assert isinstance(result["underwater_duration_days"], int)

    def test_sharpe_list_input(self):
        result = sharpe_ratio([0.01, 0.02, -0.01, 0.03, 0.005])
        assert np.isfinite(result)

    def test_var_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            value_at_risk(pd.Series([0.01, -0.02, 0.03]), method="invalid")
