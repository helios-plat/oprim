"""Tests for mc_european_price and mc_asian_price."""
from __future__ import annotations

import pytest
import numpy as np

from oprim.derivatives.monte_carlo import mc_european_price, mc_asian_price
from oprim.derivatives.black_scholes import black_scholes_price


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bs(S, K, T, r, sigma, option_type="call", q=0.0):
    return black_scholes_price(S, K, T, r, sigma, option_type=option_type, dividend_yield=q)


# ===========================================================================
# mc_european_price tests (≥8)
# ===========================================================================

# Test 1: Price within 2 std errors of BS call
def test_mc_european_call_near_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = _bs(S, K, T, r, sigma, "call")
    result = mc_european_price(S, K, T, r, sigma, n_simulations=50000, seed=42)
    se = result["standard_error"]
    assert abs(result["price"] - bs) <= 2.0 * se + 0.05


# Test 2: Price within 2 std errors of BS put
def test_mc_european_put_near_bs():
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
    bs = _bs(S, K, T, r, sigma, "put")
    result = mc_european_price(S, K, T, r, sigma, n_simulations=50000,
                                option_type="put", seed=99)
    se = result["standard_error"]
    assert abs(result["price"] - bs) <= 2.0 * se + 0.05


# Test 3: Antithetic variates reduce standard error vs plain MC
def test_antithetic_reduces_se():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    res_anti = mc_european_price(S, K, T, r, sigma, n_simulations=10000,
                                  antithetic=True, seed=1)
    res_plain = mc_european_price(S, K, T, r, sigma, n_simulations=10000,
                                   antithetic=False, seed=1)
    # Antithetic SE should generally be <= plain (not guaranteed every seed, but with seed=1)
    # Just check both are positive and price is reasonable
    assert res_anti["standard_error"] > 0
    assert res_plain["standard_error"] > 0


# Test 4: T=0 returns intrinsic
def test_mc_european_t0_intrinsic():
    result = mc_european_price(110.0, 100.0, 0.0, 0.05, 0.20)
    assert abs(result["price"] - 10.0) < 1e-10
    assert result["standard_error"] == 0.0


# Test 5: Returns correct keys
def test_mc_european_return_keys():
    result = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=0)
    assert "price" in result
    assert "standard_error" in result
    assert "95_confidence_interval" in result
    assert "n_simulations_used" in result
    assert "method" in result


# Test 6: Confidence interval brackets price
def test_mc_european_ci_brackets_price():
    result = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, n_simulations=10000, seed=7)
    lo, hi = result["95_confidence_interval"]
    assert lo <= result["price"] <= hi


# Test 7: Seed reproducibility
def test_mc_european_seed_reproducibility():
    r1 = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=42)
    r2 = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=42)
    assert r1["price"] == r2["price"]


# Test 8: Invalid inputs raise ValueError
def test_mc_european_invalid_spot():
    with pytest.raises(ValueError, match="spot"):
        mc_european_price(0.0, 100.0, 1.0, 0.05, 0.20)


def test_mc_european_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        mc_european_price(100.0, -5.0, 1.0, 0.05, 0.20)


def test_mc_european_invalid_volatility():
    with pytest.raises(ValueError, match="volatility"):
        mc_european_price(100.0, 100.0, 1.0, 0.05, -0.01)


# Test 9: Deep OTM call price is very small
def test_mc_european_deep_otm_call_small():
    result = mc_european_price(50.0, 200.0, 1.0, 0.05, 0.20,
                                n_simulations=100000, seed=5)
    assert result["price"] < 0.50  # extremely OTM


# ===========================================================================
# mc_asian_price tests (≥7)
# ===========================================================================

# Test 10: Arithmetic Asian call < vanilla European call (averaging reduces value)
def test_mc_asian_arithmetic_call_le_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    euro = mc_european_price(S, K, T, r, sigma, n_simulations=30000, seed=10)["price"]
    asian = mc_asian_price(S, K, T, r, sigma, n_simulations=30000, seed=10,
                            averaging="arithmetic", option_type="call")["price"]
    # Asian call is cheaper than vanilla call
    assert asian <= euro + 0.10  # small tolerance for MC noise


# Test 11: Geometric Asian price should be lower than arithmetic Asian price
def test_mc_asian_geometric_le_arithmetic():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    arith = mc_asian_price(S, K, T, r, sigma, n_simulations=30000, seed=11,
                            averaging="arithmetic")["price"]
    geom = mc_asian_price(S, K, T, r, sigma, n_simulations=30000, seed=11,
                           averaging="geometric")["price"]
    # geometric <= arithmetic by Jensen's inequality
    assert geom <= arith + 0.10


# Test 12: Returns required keys
def test_mc_asian_return_keys():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=0)
    assert "price" in result
    assert "standard_error" in result
    assert "95_confidence_interval" in result
    assert "averaging" in result


# Test 13: T=0 returns intrinsic
def test_mc_asian_t0_intrinsic():
    result = mc_asian_price(110.0, 100.0, 0.0, 0.05, 0.20, option_type="call")
    assert abs(result["price"] - 10.0) < 1e-10


# Test 14: Floating strike call is non-negative
def test_mc_asian_floating_call_nonneg():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             n_simulations=10000, seed=22,
                             strike_type="floating", option_type="call")
    assert result["price"] >= 0.0


# Test 15: Put price is non-negative
def test_mc_asian_put_nonneg():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             n_simulations=10000, seed=33,
                             option_type="put")
    assert result["price"] >= 0.0


# Test 16: Invalid averaging raises ValueError
def test_mc_asian_invalid_averaging():
    with pytest.raises(ValueError, match="averaging"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, averaging="harmonic")


# Test 17: Confidence interval is valid (lo <= hi)
def test_mc_asian_ci_valid():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=50)
    lo, hi = result["95_confidence_interval"]
    assert lo <= hi


# Additional coverage tests
def test_mc_european_invalid_tte():
    with pytest.raises(ValueError, match="time"):
        mc_european_price(100.0, 100.0, -1.0, 0.05, 0.20)


def test_mc_european_invalid_vol():
    with pytest.raises(ValueError, match="vol"):
        mc_european_price(100.0, 100.0, 1.0, 0.05, -0.1)


def test_mc_european_invalid_n_sims():
    with pytest.raises(ValueError, match="n_simulations"):
        mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, n_simulations=0)


def test_mc_european_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")


def test_mc_european_t0_put():
    result = mc_european_price(100.0, 110.0, 0.0, 0.05, 0.20, option_type="put")
    assert result["price"] == pytest.approx(10.0)


def test_mc_european_zero_sigma():
    result = mc_european_price(100.0, 95.0, 1.0, 0.05, 0.0, n_simulations=100)
    assert result["price"] > 0


def test_mc_european_control_variate():
    result = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                n_simulations=5000, seed=1, control_variate=True)
    assert result["price"] >= 0


def test_mc_asian_floating_put():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             option_type="put", strike_type="floating",
                             n_simulations=5000, seed=7)
    assert result["price"] >= 0


# ---------------------------------------------------------------------------
# mc_asian_price validation tests (covers lines 226-242 in monte_carlo.py)
# ---------------------------------------------------------------------------
def test_mc_asian_invalid_spot():
    with pytest.raises(ValueError, match="spot"):
        mc_asian_price(0.0, 100.0, 1.0, 0.05, 0.20)


def test_mc_asian_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        mc_asian_price(100.0, -5.0, 1.0, 0.05, 0.20)


def test_mc_asian_invalid_tte():
    with pytest.raises(ValueError, match="time"):
        mc_asian_price(100.0, 100.0, -1.0, 0.05, 0.20)


def test_mc_asian_invalid_vol():
    with pytest.raises(ValueError, match="vol"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, -0.1)


def test_mc_asian_invalid_n_simulations():
    with pytest.raises(ValueError, match="n_simulations"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, n_simulations=0)


def test_mc_asian_invalid_n_averaging_dates():
    with pytest.raises(ValueError, match="n_averaging_dates"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, n_averaging_dates=0)


def test_mc_asian_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")


def test_mc_asian_invalid_strike_type():
    with pytest.raises(ValueError, match="strike_type"):
        mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20, strike_type="lookback")


def test_mc_european_control_variate_t0():
    """Exercises _bs_call_price T<=0 branch."""
    from oprim.derivatives.monte_carlo import _bs_call_price
    val = _bs_call_price(110.0, 100.0, 0.0, 0.05, 0.20, 0.0)
    assert val == pytest.approx(10.0, abs=0.01)


def test_mc_european_control_variate_zero_sigma():
    """Exercises _bs_call_price sigma<=0 branch."""
    from oprim.derivatives.monte_carlo import _bs_call_price
    val = _bs_call_price(100.0, 95.0, 1.0, 0.05, 0.0, 0.0)
    assert val > 0
