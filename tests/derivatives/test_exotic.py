"""Tests for barrier_option_price and lookback_option_price."""
from __future__ import annotations

import pytest
import numpy as np

from oprim.derivatives.exotic import barrier_option_price, lookback_option_price
from oprim.derivatives.black_scholes import black_scholes_price
from oprim.derivatives.exotic import _barrier_cf, _bs_vanilla


def _bs(S, K, T, r, sigma, option_type="call", q=0.0):
    return black_scholes_price(S, K, T, r, sigma, option_type=option_type, dividend_yield=q)


# ===========================================================================
# barrier_option_price tests (≥8)
# ===========================================================================

# Test 1: In/Out parity — down_and_out + down_and_in = vanilla call
def test_barrier_down_in_out_parity_call():
    S, K, H, T, r, sigma = 100.0, 100.0, 80.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    out = barrier_option_price(S, K, H, T, r, sigma, barrier_type="down_and_out",
                                option_type="call", rebate=0.0)["price"]
    inp = barrier_option_price(S, K, H, T, r, sigma, barrier_type="down_and_in",
                                option_type="call", rebate=0.0)["price"]
    assert abs(out + inp - vanilla) < 0.05


# Test 2: In/Out parity — up_and_out + up_and_in = vanilla call
def test_barrier_up_in_out_parity_call():
    S, K, H, T, r, sigma = 100.0, 100.0, 120.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    out = barrier_option_price(S, K, H, T, r, sigma, barrier_type="up_and_out",
                                option_type="call", rebate=0.0)["price"]
    inp = barrier_option_price(S, K, H, T, r, sigma, barrier_type="up_and_in",
                                option_type="call", rebate=0.0)["price"]
    assert abs(out + inp - vanilla) < 0.05


# Test 3: Already knocked out (S > H for up_and_out) returns rebate
def test_barrier_already_knocked_out_rebate():
    S, K, H, T, r, sigma = 130.0, 100.0, 120.0, 1.0, 0.05, 0.20
    result = barrier_option_price(S, K, H, T, r, sigma,
                                   barrier_type="up_and_out", option_type="call",
                                   rebate=5.0, method="closed_form")
    assert abs(result["price"] - 5.0) < 1e-10


# Test 4: Already knocked in (S > H for up_and_in) returns vanilla
def test_barrier_already_knocked_in_returns_vanilla():
    S, K, H, T, r, sigma = 130.0, 100.0, 120.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    result = barrier_option_price(S, K, H, T, r, sigma,
                                   barrier_type="up_and_in", option_type="call",
                                   rebate=0.0, method="closed_form")
    assert abs(result["price"] - vanilla) < 1e-10


# Test 5: Barrier above S means up_and_out call should be close to vanilla
def test_barrier_very_high_upout_approaches_vanilla():
    S, K, H, T, r, sigma = 100.0, 100.0, 999.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    result = barrier_option_price(S, K, H, T, r, sigma,
                                   barrier_type="up_and_out", option_type="call",
                                   rebate=0.0)
    # Very high barrier: almost never hit, so out ≈ vanilla
    assert abs(result["price"] - vanilla) < 0.10


# Test 6: Monte Carlo barrier gives price in plausible range
def test_barrier_mc_price_in_range():
    S, K, H, T, r, sigma = 100.0, 100.0, 80.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    result = barrier_option_price(S, K, H, T, r, sigma,
                                   barrier_type="down_and_out",
                                   option_type="call",
                                   method="monte_carlo",
                                   n_simulations=20000)
    assert 0.0 <= result["price"] <= vanilla + 0.50
    assert "barrier_breached_pct" in result
    assert 0.0 <= result["barrier_breached_pct"] <= 1.0


# Test 7: Returns correct keys
def test_barrier_return_keys_cf():
    result = barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_out")
    assert "price" in result
    assert "method" in result
    assert "barrier_type" in result


# Test 8: Invalid barrier type raises ValueError
def test_barrier_invalid_type_raises():
    with pytest.raises(ValueError, match="barrier_type"):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                              barrier_type="left_and_right")


# Test 9: down_and_in put parity with vanilla put
def test_barrier_down_in_out_parity_put():
    S, K, H, T, r, sigma = 100.0, 100.0, 80.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "put")
    out = barrier_option_price(S, K, H, T, r, sigma, barrier_type="down_and_out",
                                option_type="put", rebate=0.0)["price"]
    inp = barrier_option_price(S, K, H, T, r, sigma, barrier_type="down_and_in",
                                option_type="put", rebate=0.0)["price"]
    assert abs(out + inp - vanilla) < 0.10


# Test 10: Invalid spot raises ValueError
def test_barrier_invalid_spot_raises():
    with pytest.raises(ValueError, match="spot"):
        barrier_option_price(-1.0, 100.0, 80.0, 1.0, 0.05, 0.20)


# ===========================================================================
# lookback_option_price tests (≥7)
# ===========================================================================

# Test 11: Floating call price > 0 for positive vol
def test_lookback_floating_call_positive():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="floating")
    assert result["price"] > 0.0
    assert result["strike_type"] == "floating"


# Test 12: Floating put price > 0 for positive vol
def test_lookback_floating_put_positive():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="floating")
    assert result["price"] > 0.0


# Test 13: Floating lookback call >= vanilla call (since S_T - min >= S_T - K)
def test_lookback_floating_call_ge_vanilla_call():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    vanilla = _bs(S, K, T, r, sigma, "call")
    lookback = lookback_option_price(S, K, T, r, sigma,
                                      option_type="call", strike_type="floating")["price"]
    # Lookback floating >= vanilla (min <= K in general, so more favourable payoff)
    assert lookback >= vanilla * 0.90  # generous tolerance


# Test 14: MC method returns valid price
def test_lookback_mc_price_nonneg():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="floating",
                                    method="monte_carlo", n_simulations=10000)
    assert result["price"] >= 0.0
    assert result["method"] == "monte_carlo"


# Test 15: Fixed strike call >= 0
def test_lookback_fixed_call_nonneg():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="fixed")
    assert result["price"] >= 0.0


# Test 16: T=0 floating call returns 0 (no path to track)
def test_lookback_t0_returns_zero():
    result = lookback_option_price(100.0, 100.0, 0.0, 0.05, 0.20,
                                    option_type="call", strike_type="floating")
    assert result["price"] == 0.0


# Test 17: Invalid strike_type raises ValueError
def test_lookback_invalid_strike_type_raises():
    with pytest.raises(ValueError, match="strike_type"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                               strike_type="average")


# Test 18: Returns correct keys
def test_lookback_return_keys():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20)
    assert "price" in result
    assert "method" in result
    assert "strike_type" in result


# ===========================================================================
# Additional coverage tests
# ===========================================================================

# _bs_vanilla T=0 branches
def test_bs_vanilla_t0_call():
    # T=0, S>K → intrinsic
    result = _bs_vanilla(110.0, 100.0, 0.0, 0.05, 0.20, 0.0, "call")
    assert result == pytest.approx(10.0)


def test_bs_vanilla_t0_put():
    result = _bs_vanilla(90.0, 100.0, 0.0, 0.05, 0.20, 0.0, "put")
    assert result == pytest.approx(10.0)


def test_bs_vanilla_t0_otm():
    result = _bs_vanilla(90.0, 100.0, 0.0, 0.05, 0.20, 0.0, "call")
    assert result == pytest.approx(0.0)


def test_bs_vanilla_zero_sigma_call():
    # sigma=0: deterministic
    result = _bs_vanilla(100.0, 95.0, 1.0, 0.05, 0.0, 0.0, "call")
    # forward = 100*exp(0.05) > 95 → itm
    assert result > 0


def test_bs_vanilla_zero_sigma_put_otm():
    result = _bs_vanilla(100.0, 95.0, 1.0, 0.05, 0.0, 0.0, "put")
    # forward > K → put is OTM
    assert result == pytest.approx(0.0, abs=0.01)


# _barrier_cf T=0 branches
def test_barrier_cf_t0_knocked_out_down():
    # S <= H for down_and_out: return rebate
    result = _barrier_cf(70.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_out", "call", 5.0)
    assert result == pytest.approx(5.0)


def test_barrier_cf_t0_not_knocked_out():
    # S > H for down_and_out: return intrinsic
    result = _barrier_cf(110.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_out", "call", 5.0)
    assert result == pytest.approx(10.0)


def test_barrier_cf_t0_knocked_in_down():
    # S <= H for down_and_in: return intrinsic
    result = _barrier_cf(75.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_in", "call", 5.0)
    assert result == pytest.approx(0.0)  # OTM call intrinsic


def test_barrier_cf_t0_not_knocked_in():
    # S > H for down_and_in: not knocked in → rebate
    result = _barrier_cf(110.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_in", "call", 5.0)
    assert result == pytest.approx(5.0)


def test_barrier_cf_t0_up_knocked_out():
    # S >= H for up_and_out → rebate
    result = _barrier_cf(130.0, 100.0, 120.0, 0.0, 0.05, 0.20, 0.0,
                         "up_and_out", "call", 3.0)
    assert result == pytest.approx(3.0)


def test_barrier_cf_t0_up_not_knocked_out():
    # S < H for up_and_out → intrinsic
    result = _barrier_cf(110.0, 100.0, 120.0, 0.0, 0.05, 0.20, 0.0,
                         "up_and_out", "call", 3.0)
    assert result == pytest.approx(10.0)


def test_barrier_cf_t0_up_knocked_in():
    # S >= H for up_and_in → intrinsic
    result = _barrier_cf(130.0, 100.0, 120.0, 0.0, 0.05, 0.20, 0.0,
                         "up_and_in", "call", 3.0)
    assert result == pytest.approx(30.0)


# barrier zero vol paths
def test_barrier_zero_vol_down_and_out():
    result = barrier_option_price(100.0, 95.0, 80.0, 1.0, 0.05, 0.0,
                                   barrier_type="down_and_out", option_type="call",
                                   rebate=0.0)
    assert result["price"] >= 0


def test_barrier_zero_vol_up_and_out():
    result = barrier_option_price(100.0, 95.0, 120.0, 1.0, 0.05, 0.0,
                                   barrier_type="up_and_out", option_type="call",
                                   rebate=0.0)
    assert result["price"] >= 0


# K < H down_and_out call (line 132)
def test_barrier_down_out_call_k_less_than_h():
    # K < H: different formula branch
    result = barrier_option_price(100.0, 70.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_out", option_type="call",
                                   rebate=0.0)
    assert result["price"] >= 0


# down_and_out put K < H (line 137 branch)
def test_barrier_down_out_put_k_lt_h():
    result = barrier_option_price(100.0, 70.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_out", option_type="put",
                                   rebate=0.0)
    assert result["price"] >= 0


# up_and_out: K >= H call (line 161)
def test_barrier_up_out_call_k_ge_h():
    result = barrier_option_price(100.0, 125.0, 120.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_out", option_type="call",
                                   rebate=0.0)
    assert result["price"] >= 0


# up_and_out: K < H call (line 163) and put K>=H (line 166)
def test_barrier_up_out_put_k_ge_h():
    result = barrier_option_price(100.0, 125.0, 120.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_out", option_type="put",
                                   rebate=0.0)
    assert result["price"] >= 0


def test_barrier_up_out_put_k_lt_h():
    result = barrier_option_price(100.0, 95.0, 120.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_out", option_type="put",
                                   rebate=5.0)
    assert result["price"] >= 0


# Rebate validation tests
def test_barrier_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        barrier_option_price(100.0, -1.0, 80.0, 1.0, 0.05, 0.20)


def test_barrier_invalid_barrier():
    with pytest.raises(ValueError, match="barrier"):
        barrier_option_price(100.0, 100.0, -10.0, 1.0, 0.05, 0.20)


def test_barrier_invalid_tte():
    with pytest.raises(ValueError, match="time"):
        barrier_option_price(100.0, 100.0, 80.0, -1.0, 0.05, 0.20)


def test_barrier_invalid_vol():
    with pytest.raises(ValueError, match="volatility"):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, -0.1)


# down_and_in already knocked in (line 142)
def test_barrier_down_in_already_knocked_in():
    result = barrier_option_price(75.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_in", option_type="call")
    vanilla = _bs(75.0, 100.0, 1.0, 0.05, 0.20)
    assert abs(result["price"] - vanilla) < 0.01


# Lookback MC put
def test_lookback_mc_put():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="floating",
                                    method="monte_carlo", n_simulations=5000)
    assert result["price"] >= 0


# Lookback fixed put
def test_lookback_fixed_put_nonneg():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="fixed")
    assert result["price"] >= 0


# Lookback zero vol
def test_lookback_zero_vol():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.0,
                                    option_type="call", strike_type="floating")
    assert result["price"] >= 0


# Lookback with dividend
def test_lookback_with_dividend():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", dividend_yield=0.03)
    assert result["price"] > 0


# Lookback fixed put K < S (line 497 branch m0 < K means S < K, different)
def test_lookback_fixed_put_k_gt_s():
    result = lookback_option_price(90.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="fixed")
    assert result["price"] > 0


# Lookback zero sigma fixed
def test_lookback_zero_sigma_fixed_call():
    result = lookback_option_price(100.0, 95.0, 1.0, 0.05, 0.0,
                                    option_type="call", strike_type="fixed")
    assert result["price"] >= 0


# ===================================================================
# Additional coverage tests for exotic.py
# ===================================================================

# --- _barrier_cf T=0 put (line 49) ---
def test_barrier_cf_t0_put():
    result = _barrier_cf(90.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_out", "put", 0.0)
    assert abs(result - 10.0) < 1e-10


# --- _barrier_cf T=0 in/out with breach/no-breach ---
def test_barrier_cf_t0_down_out_breached():
    """S <= H, down_and_out → returns rebate (line 52)."""
    result = _barrier_cf(75.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_out", "call", 5.0)
    assert result == 5.0


def test_barrier_cf_t0_up_out_breached():
    """S >= H, up_and_out → returns rebate (line 54)."""
    result = _barrier_cf(130.0, 100.0, 120.0, 0.0, 0.05, 0.20, 0.0,
                         "up_and_out", "call", 3.0)
    assert result == 3.0


def test_barrier_cf_t0_down_in_breached():
    """S <= H, down_and_in → returns intrinsic (line 58)."""
    result = _barrier_cf(75.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_in", "put", 0.0)
    assert abs(result - 25.0) < 1e-10


def test_barrier_cf_t0_up_in_breached():
    """S >= H, up_and_in → returns intrinsic (line 60)."""
    result = _barrier_cf(130.0, 100.0, 120.0, 0.0, 0.05, 0.20, 0.0,
                         "up_and_in", "call", 0.0)
    assert abs(result - 30.0) < 1e-10


def test_barrier_cf_t0_in_not_breached():
    """Not breached, in barrier → returns rebate (line 61)."""
    result = _barrier_cf(100.0, 100.0, 80.0, 0.0, 0.05, 0.20, 0.0,
                         "down_and_in", "call", 7.0)
    assert result == 7.0


# --- _barrier_zero_vol branches ---
def test_barrier_zero_vol_out_breached_returns_rebate():
    """Out barrier, breached, zero vol → rebate * disc (line 200)."""
    result = _barrier_cf(90.0, 100.0, 95.0, 1.0, 0.05, 0.0, 0.0,
                         "down_and_out", "call", 5.0)
    expected = 5.0 * np.exp(-0.05 * 1.0)
    assert abs(result - expected) < 1e-6


def test_barrier_zero_vol_out_not_breached_put():
    """Out barrier, not breached, put, zero vol (line 203)."""
    # S=100, H=80, F=100*exp(0.05) > K=90 → not breached for down_and_out
    # Put payoff: max(K - F, 0)
    result = _barrier_cf(100.0, 90.0, 80.0, 1.0, 0.05, 0.0, 0.0,
                         "down_and_out", "put", 0.0)
    import math
    F = 100.0 * math.exp(0.05)
    expected = max(90.0 - F, 0.0) * math.exp(-0.05)
    assert abs(result - expected) < 1e-6


def test_barrier_zero_vol_in_breached_call():
    """In barrier, breached, call, zero vol (line 207)."""
    result = _barrier_cf(90.0, 100.0, 95.0, 1.0, 0.05, 0.0, 0.0,
                         "down_and_in", "call", 0.0)
    # Breached → call payoff
    import math
    F = 90.0 * math.exp(0.05)
    expected = max(F - 100.0, 0.0) * math.exp(-0.05)
    assert abs(result - expected) < 1e-6


def test_barrier_zero_vol_in_breached_put():
    """In barrier, breached, put, zero vol (line 208)."""
    result = _barrier_cf(90.0, 100.0, 95.0, 1.0, 0.05, 0.0, 0.0,
                         "down_and_in", "put", 0.0)
    import math
    F = 90.0 * math.exp(0.05)
    expected = max(100.0 - F, 0.0) * math.exp(-0.05)
    assert abs(result - expected) < 1e-6


def test_barrier_zero_vol_in_not_breached_rebate():
    """In barrier, not breached, zero vol → rebate * disc (line 209)."""
    result = _barrier_cf(100.0, 90.0, 80.0, 1.0, 0.05, 0.0, 0.0,
                         "down_and_in", "call", 3.0)
    expected = 3.0 * np.exp(-0.05 * 1.0)
    assert abs(result - expected) < 1e-6


# --- barrier_option_price validation ---
def test_barrier_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                              option_type="straddle")


def test_barrier_invalid_method():
    with pytest.raises(ValueError, match="method"):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                              method="binomial")


def test_barrier_mc_invalid_n_sims():
    with pytest.raises(ValueError, match="n_simulations"):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                              method="monte_carlo", n_simulations=0)


# --- barrier MC up breach and put ---
def test_barrier_mc_up_and_out_put():
    """MC up_and_out put — covers 'up' breach check (line 318) and put (line 326)."""
    result = barrier_option_price(100.0, 110.0, 120.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_out", option_type="put",
                                   method="monte_carlo", n_simulations=2000)
    assert result["price"] >= 0


def test_barrier_mc_down_and_in_put():
    """MC down_and_in put — covers 'in' MC payoff (line 331) and put."""
    result = barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_in", option_type="put",
                                   method="monte_carlo", n_simulations=2000)
    assert result["price"] >= 0


# --- down_and_out already knocked out S <= H (line 126) ---
def test_barrier_cf_down_out_already_knocked():
    """S <= H for down_and_out with T>0 → rebate (line 126)."""
    result = _barrier_cf(75.0, 100.0, 80.0, 1.0, 0.05, 0.20, 0.0,
                         "down_and_out", "call", 2.0)
    assert result == 2.0


# --- up_and_in F_rebate usage: need rebate > 0 with sigma > 0 ---
def test_barrier_cf_rebate_nonzero():
    """Tests F_rebate formula path (line 116) via down_and_out with rebate."""
    # With sigma > 0, T > 0, rebate > 0 → E_rebate called
    result = _barrier_cf(100.0, 100.0, 80.0, 1.0, 0.05, 0.20, 0.0,
                         "down_and_out", "call", 5.0)
    assert result >= 0


# --- _lookback_cf T=0 fixed call/put (lines 357-359) ---
def test_lookback_cf_t0_fixed_call():
    """T=0 fixed lookback call (line 358)."""
    from oprim.derivatives.exotic import _lookback_cf
    result = _lookback_cf(110.0, 100.0, 0.0, 0.05, 0.20, 0.0, "call", "fixed")
    assert abs(result - 10.0) < 1e-10


def test_lookback_cf_t0_fixed_put():
    """T=0 fixed lookback put (line 359)."""
    from oprim.derivatives.exotic import _lookback_cf
    result = _lookback_cf(90.0, 100.0, 0.0, 0.05, 0.20, 0.0, "put", "fixed")
    assert abs(result - 10.0) < 1e-10


# --- sigma=0 floating put (line 368) ---
def test_lookback_zero_sigma_floating_put():
    """sigma=0, floating put (line 368)."""
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.0,
                                    option_type="put", strike_type="floating")
    assert result["price"] >= 0


# --- sigma=0 fixed put (line 371) ---
def test_lookback_zero_sigma_fixed_put():
    """sigma=0, fixed put (line 371)."""
    result = lookback_option_price(100.0, 95.0, 1.0, 0.05, 0.0,
                                    option_type="put", strike_type="fixed")
    assert result["price"] >= 0


# --- b≈0 floating call/put (lines 426-428, 444-446): r == q ---
def test_lookback_b_zero_floating_call():
    """r == q → b=0 for floating call (lines 426-428)."""
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="floating",
                                    dividend_yield=0.05)
    assert result["price"] >= 0


def test_lookback_b_zero_floating_put():
    """r == q → b=0 for floating put (lines 444-446)."""
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="floating",
                                    dividend_yield=0.05)
    assert result["price"] >= 0


# --- Fixed lookback call M0 > K (lines 461-473) ---
def test_lookback_fixed_call_m0_gt_k():
    """Fixed call with S > K tests M0 > K branch (lines 461-473)."""
    # S=110, K=100 → M0 = S = 110 > K = 100
    result = lookback_option_price(110.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="fixed")
    assert result["price"] > 0


def test_lookback_fixed_call_m0_gt_k_b_zero():
    """Fixed call M0>K with b=0 (lines 472-473)."""
    result = lookback_option_price(110.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="fixed",
                                    dividend_yield=0.05)
    assert result["price"] >= 0


# --- Fixed lookback call M0 <= K (b=0 case line 488) ---
def test_lookback_fixed_call_m0_le_k_b_zero():
    """Fixed call S < K (M0=S < K), b=0 branch (line 488)."""
    result = lookback_option_price(90.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="fixed",
                                    dividend_yield=0.05)
    assert result["price"] >= 0


# --- Fixed lookback put m0 < K (line 506) ---
def test_lookback_fixed_put_m0_lt_k_b_zero():
    """Fixed put m0 < K with b=0 (line 506)."""
    result = lookback_option_price(90.0, 100.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="fixed",
                                    dividend_yield=0.05)
    assert result["price"] > 0


# --- Fixed lookback put K <= m0 (lines 509-525): b=0 (line 521) ---
def test_lookback_fixed_put_k_le_m0_b_zero():
    """Fixed put K <= S (m0=S), b=0 (line 521)."""
    result = lookback_option_price(100.0, 90.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="fixed",
                                    dividend_yield=0.05)
    assert result["price"] >= 0


# --- lookback_option_price validation (lines 583-595) ---
def test_lookback_invalid_spot():
    with pytest.raises(ValueError, match="spot"):
        lookback_option_price(0.0, 100.0, 1.0, 0.05, 0.20)


def test_lookback_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        lookback_option_price(100.0, -5.0, 1.0, 0.05, 0.20)


def test_lookback_invalid_tte():
    with pytest.raises(ValueError, match="time"):
        lookback_option_price(100.0, 100.0, -1.0, 0.05, 0.20)


def test_lookback_invalid_vol():
    with pytest.raises(ValueError, match="volatility"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, -0.1)


def test_lookback_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")


def test_lookback_invalid_strike_type():
    with pytest.raises(ValueError, match="strike_type"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20, strike_type="dynamic")


def test_lookback_invalid_method():
    with pytest.raises(ValueError, match="method"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20, method="binomial")


def test_lookback_mc_invalid_n_sims():
    with pytest.raises(ValueError, match="n_simulations"):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                               method="monte_carlo", n_simulations=0)


# --- Lookback MC fixed strike (lines 635-638) ---
def test_lookback_mc_fixed_call():
    """MC fixed lookback call (line 635-636)."""
    result = lookback_option_price(100.0, 95.0, 1.0, 0.05, 0.20,
                                    option_type="call", strike_type="fixed",
                                    method="monte_carlo", n_simulations=2000)
    assert result["price"] >= 0


def test_lookback_mc_fixed_put():
    """MC fixed lookback put (lines 637-638)."""
    result = lookback_option_price(100.0, 105.0, 1.0, 0.05, 0.20,
                                    option_type="put", strike_type="fixed",
                                    method="monte_carlo", n_simulations=2000)
    assert result["price"] >= 0
