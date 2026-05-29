"""Tests for binomial_tree_price."""
from __future__ import annotations

import math
import pytest
import numpy as np

from oprim.derivatives.binomial_tree import binomial_tree_price
from oprim.derivatives.black_scholes import black_scholes_price


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bs(S, K, T, r, sigma, option_type="call", q=0.0):
    return black_scholes_price(S, K, T, r, sigma, option_type=option_type, dividend_yield=q)


# ---------------------------------------------------------------------------
# Test 1: European call converges to BS (CRR, n=500, within 1%)
# ---------------------------------------------------------------------------
def test_european_call_crr_converges_to_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = _bs(S, K, T, r, sigma, "call")
    result = binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="call",
                                  exercise="european", method="crr")
    assert abs(result["price"] - bs_price) / bs_price < 0.01
    assert result["method"] == "crr"
    assert result["n_steps"] == 500


# ---------------------------------------------------------------------------
# Test 2: European put converges to BS (CRR, n=500, within 1%)
# ---------------------------------------------------------------------------
def test_european_put_crr_converges_to_bs():
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.25
    bs_price = _bs(S, K, T, r, sigma, "put")
    result = binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="put",
                                  exercise="european", method="crr")
    assert abs(result["price"] - bs_price) / (bs_price + 1e-6) < 0.01


# ---------------------------------------------------------------------------
# Test 3: Jarrow-Rudd European call converges to BS within 1%
# ---------------------------------------------------------------------------
def test_european_call_jr_converges_to_bs():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs_price = _bs(S, K, T, r, sigma, "call")
    result = binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="call",
                                  exercise="european", method="jarrow_rudd")
    assert abs(result["price"] - bs_price) / bs_price < 0.01
    assert result["method"] == "jarrow_rudd"


# ---------------------------------------------------------------------------
# Test 4: American put >= European put (early exercise premium)
# ---------------------------------------------------------------------------
def test_american_put_ge_european_put():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    amer = binomial_tree_price(S, K, T, r, sigma, n_steps=100, option_type="put",
                                exercise="american")
    euro = binomial_tree_price(S, K, T, r, sigma, n_steps=100, option_type="put",
                                exercise="european")
    assert amer["price"] >= euro["price"] - 1e-10


# ---------------------------------------------------------------------------
# Test 5: American call on non-dividend-paying stock equals European call
# ---------------------------------------------------------------------------
def test_american_call_no_div_equals_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    amer = binomial_tree_price(S, K, T, r, sigma, n_steps=200, option_type="call",
                                exercise="american", dividend_yield=0.0)
    euro = binomial_tree_price(S, K, T, r, sigma, n_steps=200, option_type="call",
                                exercise="european", dividend_yield=0.0)
    assert abs(amer["price"] - euro["price"]) < 0.01


# ---------------------------------------------------------------------------
# Test 6: T=0 returns intrinsic value
# ---------------------------------------------------------------------------
def test_t_zero_returns_intrinsic():
    result_call = binomial_tree_price(110.0, 100.0, 0.0, 0.05, 0.20, option_type="call")
    assert abs(result_call["price"] - 10.0) < 1e-10
    result_put = binomial_tree_price(90.0, 100.0, 0.0, 0.05, 0.20, option_type="put")
    assert abs(result_put["price"] - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 7: American put early exercise boundary is returned
# ---------------------------------------------------------------------------
def test_american_put_early_exercise_boundary():
    result = binomial_tree_price(100.0, 110.0, 1.0, 0.05, 0.20, n_steps=50,
                                  option_type="put", exercise="american")
    assert "early_exercise_boundary" in result
    assert len(result["early_exercise_boundary"]) == 50


# ---------------------------------------------------------------------------
# Test 8: Invalid inputs raise ValueError
# ---------------------------------------------------------------------------
def test_invalid_spot_raises():
    with pytest.raises(ValueError, match="spot"):
        binomial_tree_price(0.0, 100.0, 1.0, 0.05, 0.20)

def test_invalid_strike_raises():
    with pytest.raises(ValueError, match="strike"):
        binomial_tree_price(100.0, -1.0, 1.0, 0.05, 0.20)

def test_invalid_n_steps_raises():
    with pytest.raises(ValueError, match="n_steps"):
        binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.20, n_steps=0)

def test_invalid_option_type_raises():
    with pytest.raises(ValueError, match="option_type"):
        binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")

def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.20, method="trinomial")


# ---------------------------------------------------------------------------
# Test 9: Deep ITM call price approaches intrinsic lower bound
# ---------------------------------------------------------------------------
def test_deep_itm_call_price_reasonable():
    S, K = 200.0, 100.0
    result = binomial_tree_price(S, K, 1.0, 0.05, 0.20, n_steps=100,
                                  option_type="call", exercise="european")
    # Must be >= intrinsic discounted
    assert result["price"] >= (S - K) * math.exp(-0.05 * 1.0) * 0.99


# ---------------------------------------------------------------------------
# Test 10: Put-call parity approximately holds for European (tree)
# ---------------------------------------------------------------------------
def test_put_call_parity_european_tree():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    call = binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="call",
                                exercise="european")["price"]
    put = binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="put",
                               exercise="european")["price"]
    # C - P = S - K*exp(-rT)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 0.20  # within 20 cents for n=500


# ---------------------------------------------------------------------------
# Test 11: Dividend yield reduces call price
# ---------------------------------------------------------------------------
def test_dividend_yield_reduces_call():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    no_div = binomial_tree_price(S, K, T, r, sigma, n_steps=100, option_type="call",
                                  dividend_yield=0.0)["price"]
    with_div = binomial_tree_price(S, K, T, r, sigma, n_steps=100, option_type="call",
                                    dividend_yield=0.05)["price"]
    assert no_div > with_div


# Additional coverage tests
def test_bt_invalid_tte():
    with pytest.raises(ValueError, match="time"):
        binomial_tree_price(100.0, 100.0, -1.0, 0.05, 0.20)


def test_bt_invalid_vol():
    with pytest.raises(ValueError, match="vol"):
        binomial_tree_price(100.0, 100.0, 1.0, 0.05, -0.1)


def test_bt_invalid_exercise():
    with pytest.raises(ValueError, match="exercise"):
        binomial_tree_price(100.0, 100.0, 1.0, 0.05, 0.20, exercise="bermudan")


def test_bt_t0_american_put():
    result = binomial_tree_price(90.0, 100.0, 0.0, 0.05, 0.20,
                                  option_type="put", exercise="american")
    assert result["price"] == pytest.approx(10.0)
    assert "early_exercise_boundary" in result


def test_bt_zero_sigma_crr():
    result = binomial_tree_price(100.0, 95.0, 1.0, 0.05, 0.0, n_steps=50)
    assert result["price"] > 0


def test_bt_zero_sigma_jarrow_rudd():
    result = binomial_tree_price(100.0, 95.0, 1.0, 0.05, 0.0, n_steps=50,
                                  method="jarrow_rudd")
    assert result["price"] > 0
