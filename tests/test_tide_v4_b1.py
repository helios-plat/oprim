import pytest
from oprim._exceptions import OprimError
from oprim.kdj import kdj
from oprim.limit_status_calc import limit_status_calc
from oprim.beneish_m_score import beneish_m_score, BeneishInput
from oprim.dupont_decomposition import dupont_decomposition
from oprim.dcf_valuation import dcf_valuation

# --- KDJ Tests ---
def test_kdj_normal():
    high = [10, 11, 12, 11, 10, 9, 10, 11, 12]
    low = [9, 10, 11, 10, 9, 8, 9, 10, 11]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 11.5]
    res = kdj(high=high, low=low, close=close, n=3)
    assert len(res.k) == len(close)
    assert len(res.d) == len(close)
    assert len(res.j) == len(close)

def test_kdj_mismatch_length():
    with pytest.raises(OprimError):
        kdj(high=[1, 2], low=[1], close=[1, 2])

def test_kdj_too_short():
    with pytest.raises(OprimError):
        kdj(high=[1, 2], low=[1, 2], close=[1, 2], n=5)

def test_kdj_flat_price():
    # rng = 0, should return 50
    high = [10] * 10
    low = [10] * 10
    close = [10] * 10
    res = kdj(high=high, low=low, close=close, n=5)
    assert res.k[-1] == 50.0

def test_kdj_known_values():
    # Simple manual check for n=1
    high = [10, 20]
    low = [5, 15]
    close = [7.5, 17.5]
    # t=0: lo=5, hi=10, rng=5, rsv=(7.5-5)/5*100=50. k = (2/3)*50 + (1/3)*50 = 50. d=50. j=50.
    # t=1: lo=15, hi=20, rng=5, rsv=(17.5-15)/5*100=50. k = (2/3)*50 + (1/3)*50 = 50. d=50. j=50.
    res = kdj(high=high, low=low, close=close, n=1)
    assert res.j[-1] == 50.0

# --- Limit Status Tests ---
def test_limit_status_normal():
    close = [10, 11, 12.1, 13.31, 12, 10.8]
    # Changes: +10%, +10%, +10%, -9.8%, -10%
    res = limit_status_calc(close=close, limit_pct=0.10, lookback=5)
    assert res.recent == ["limit_up", "limit_up", "limit_up", "normal", "limit_down"]

def test_limit_status_insufficient():
    with pytest.raises(OprimError):
        limit_status_calc(close=[10, 11], lookback=5)

def test_limit_status_no_change():
    close = [10, 10, 10]
    res = limit_status_calc(close=close, limit_pct=0.10, lookback=2)
    assert res.recent == ["normal", "normal"]

def test_limit_status_varying_pct():
    close = [10, 12] # +20%
    res_10 = limit_status_calc(close=close, limit_pct=0.10, lookback=1)
    assert res_10.recent == ["limit_up"]
    res_30 = limit_status_calc(close=close, limit_pct=0.30, lookback=1)
    assert res_30.recent == ["normal"]

def test_limit_status_edge_case():
    close = [10, 11] # Exact 10%
    res = limit_status_calc(close=close, limit_pct=0.10, lookback=1)
    assert res.recent == ["limit_up"]

# --- Beneish M-Score Tests ---
def test_beneish_normal():
    curr = BeneishInput(net_profit=10, revenue=100, total_assets=500, total_liabilities=200, operating_cash_flow=8)
    prior = BeneishInput(net_profit=8, revenue=80, total_assets=450, total_liabilities=180, operating_cash_flow=6)
    res = beneish_m_score(current=curr, prior=prior)
    assert isinstance(res.m_score, float)
    assert "SGI" in res.factors

def test_beneish_fraud_risk():
    # High accruals (Net Profit >> CFO) often flags risk
    curr = BeneishInput(net_profit=50, revenue=100, total_assets=500, total_liabilities=200, operating_cash_flow=-50)
    prior = BeneishInput(net_profit=10, revenue=100, total_assets=500, total_liabilities=200, operating_cash_flow=10)
    res = beneish_m_score(current=curr, prior=prior)
    # TATA will be (50 - (-50)) / 500 = 0.2. High positive TATA increases M-score.
    assert res.m_score > -2.22

def test_beneish_division_by_zero():
    curr = BeneishInput(net_profit=10, revenue=100, total_assets=500, total_liabilities=200, operating_cash_flow=8)
    prior = BeneishInput(net_profit=10, revenue=0, total_assets=500, total_liabilities=200, operating_cash_flow=8)
    res = beneish_m_score(current=curr, prior=prior)
    assert res.factors["SGI"] == 1.0 # Default value for division by zero

def test_beneish_conservative():
    # Same data, factors should be 1.0 or 0.0
    curr = BeneishInput(net_profit=10, revenue=100, total_assets=500, total_liabilities=200, operating_cash_flow=8)
    res = beneish_m_score(current=curr, prior=curr)
    assert res.factors["SGI"] == 1.0
    assert res.factors["LVGI"] == 1.0

def test_beneish_m_score_threshold():
    # Check if we can get a low score for a healthy company
    curr = BeneishInput(net_profit=10, revenue=100, total_assets=1000, total_liabilities=100, operating_cash_flow=20)
    prior = BeneishInput(net_profit=10, revenue=110, total_assets=1000, total_liabilities=100, operating_cash_flow=20)
    res = beneish_m_score(current=curr, prior=prior)
    assert res.m_score < -2.22

# --- DuPont Tests ---
def test_dupont_normal():
    res = dupont_decomposition(net_income=10, revenue=100, total_assets=500, total_equity=200)
    assert res.roe == 0.05
    assert res.npm == 0.10
    assert res.asset_turnover == 0.20
    assert res.equity_multiplier == 2.5

def test_dupont_zero_div():
    with pytest.raises(OprimError):
        dupont_decomposition(net_income=10, revenue=0, total_assets=500, total_equity=200)

def test_dupont_zero_assets():
    with pytest.raises(OprimError):
        dupont_decomposition(net_income=10, revenue=100, total_assets=0, total_equity=200)

def test_dupont_zero_equity():
    with pytest.raises(OprimError):
        dupont_decomposition(net_income=10, revenue=100, total_assets=500, total_equity=0)

def test_dupont_scalar():
    res = dupont_decomposition(net_income=20, revenue=100, total_assets=400, total_equity=100)
    assert res.roe == 0.2

# --- DCF Tests ---
def test_dcf_normal():
    fcf = [100, 110, 121, 133, 146]
    res = dcf_valuation(free_cash_flows=fcf, discount_rate=0.10, terminal_growth_rate=0.02, shares_outstanding=1000)
    assert res.enterprise_value > sum(fcf)
    assert res.intrinsic_value_per_share == res.enterprise_value / 1000

def test_dcf_invalid_rates():
    with pytest.raises(OprimError):
        dcf_valuation(free_cash_flows=[100], discount_rate=0.05, terminal_growth_rate=0.06, shares_outstanding=100)

def test_dcf_zero_shares():
    with pytest.raises(OprimError):
        dcf_valuation(free_cash_flows=[100], shares_outstanding=0)

def test_dcf_negative_fcf():
    # DCF should still work with negative FCF
    res = dcf_valuation(free_cash_flows=[-100, 100], discount_rate=0.1, shares_outstanding=100)
    assert isinstance(res.enterprise_value, float)

def test_dcf_forecast_years():
    fcf = [100, 100, 100, 100, 100, 200, 200]
    res5 = dcf_valuation(free_cash_flows=fcf, forecast_years=5, shares_outstanding=1)
    res7 = dcf_valuation(free_cash_flows=fcf, forecast_years=7, shares_outstanding=1)
    assert res7.enterprise_value > res5.enterprise_value
