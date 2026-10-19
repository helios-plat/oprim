"""Tests for Kelly & Portfolio risk oprim."""
from oprim.risk.kelly_position import calculate_kelly_criterion, calculate_portfolio_var

def test_kelly_criterion():
    f = calculate_kelly_criterion(0.55, 1.5)
    assert 0.1 <= f <= 0.3

def test_portfolio_var():
    res = calculate_portfolio_var(10000.0, [{'size': 1, 'price': 2000.0}])
    assert res['total_exposure'] == 2000.0
    assert 'var_estimated' in res
