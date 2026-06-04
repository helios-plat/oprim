"""Tests for oprim.backtest_stat — ≥10 cases."""

from __future__ import annotations

import pytest
from oprim.backtest_stat import backtest_stat


# 1. Empty returns
def test_empty_returns():
    result = backtest_stat(returns=[])
    assert result["total_return"] == 0.0
    assert result["annualized_return"] == 0.0
    assert result["annualized_volatility"] == 0.0
    assert result["sharpe_ratio"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["win_rate"] == 0.0
    assert result["n_periods"] == 0


# 2. Single return
def test_single_return():
    result = backtest_stat(returns=[0.05])
    assert result["n_periods"] == 1
    assert result["total_return"] == pytest.approx(0.05)
    assert result["annualized_volatility"] == 0.0
    assert result["sharpe_ratio"] == 0.0
    assert result["max_drawdown"] == 0.0
    assert result["win_rate"] == 1.0


# 3. Positive series — total_return > 0
def test_positive_series():
    returns = [0.01] * 10
    result = backtest_stat(returns=returns)
    assert result["total_return"] > 0
    assert result["n_periods"] == 10
    assert result["win_rate"] == 1.0


# 4. Negative series — total_return < 0
def test_negative_series():
    returns = [-0.01] * 10
    result = backtest_stat(returns=returns)
    assert result["total_return"] < 0
    assert result["win_rate"] == 0.0


# 5. total_return correct
def test_total_return_correct():
    # (1.1)(1.2)(0.9) - 1
    returns = [0.1, 0.2, -0.1]
    expected = (1.1 * 1.2 * 0.9) - 1
    result = backtest_stat(returns=returns)
    assert result["total_return"] == pytest.approx(expected, rel=1e-9)


# 6. sharpe_ratio sign — positive returns → positive sharpe
def test_sharpe_ratio_positive():
    returns = [0.002] * 252
    result = backtest_stat(returns=returns, risk_free_rate=0.0)
    # constant positive returns → std of excess = 0, so sharpe = 0.0 (fallback)
    # with slight variation it should be positive
    returns_varied = [0.001, 0.003] * 126
    result2 = backtest_stat(returns=returns_varied, risk_free_rate=0.0)
    assert result2["sharpe_ratio"] > 0


# 7. max_drawdown ≤ 0
def test_max_drawdown_nonpositive():
    returns = [0.05, -0.10, 0.03, -0.02, 0.01]
    result = backtest_stat(returns=returns)
    assert result["max_drawdown"] <= 0.0


# 8. win_rate in [0, 1]
def test_win_rate_bounds():
    returns = [0.01, -0.02, 0.03, 0.0, -0.01]
    result = backtest_stat(returns=returns)
    assert 0.0 <= result["win_rate"] <= 1.0


# 9. n_periods correct
def test_n_periods():
    returns = [0.01, 0.02, -0.01, 0.005, 0.0]
    result = backtest_stat(returns=returns)
    assert result["n_periods"] == 5


# 10. annualized_volatility ≥ 0
def test_annualized_volatility_nonnegative():
    returns = [0.01, -0.02, 0.03, 0.0, -0.01, 0.04, -0.03]
    result = backtest_stat(returns=returns)
    assert result["annualized_volatility"] >= 0.0


# 11. all-loss series has max_drawdown < 0
def test_all_loss_series_drawdown():
    returns = [-0.02, -0.03, -0.01]
    result = backtest_stat(returns=returns)
    assert result["max_drawdown"] < 0.0


# 12. no drawdown on always-rising series
def test_no_drawdown_rising():
    returns = [0.01, 0.02, 0.005, 0.03]
    result = backtest_stat(returns=returns)
    assert result["max_drawdown"] == pytest.approx(0.0, abs=1e-10)


# 13. risk_free_rate reduces sharpe
def test_risk_free_rate_reduces_sharpe():
    returns = [0.002, 0.003, 0.001, 0.004, 0.002] * 50
    r1 = backtest_stat(returns=returns, risk_free_rate=0.0)
    r2 = backtest_stat(returns=returns, risk_free_rate=0.5)
    assert r1["sharpe_ratio"] > r2["sharpe_ratio"]


# 14. periods_per_year scales annualized metrics
def test_periods_per_year_scaling():
    returns = [0.001] * 12  # monthly-ish
    r_daily = backtest_stat(returns=returns, periods_per_year=252)
    r_monthly = backtest_stat(returns=returns, periods_per_year=12)
    # annualized return differs by scaling
    assert r_daily["annualized_return"] != pytest.approx(r_monthly["annualized_return"])
