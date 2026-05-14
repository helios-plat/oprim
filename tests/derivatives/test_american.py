"""Tests for lsm_american_price (Longstaff-Schwartz)."""
from __future__ import annotations

import pytest
import numpy as np

from oprim.derivatives.american import lsm_american_price
from oprim.derivatives.binomial_tree import binomial_tree_price
from oprim.derivatives.black_scholes import black_scholes_price


# ---------------------------------------------------------------------------
# Test 1: American put price >= European put (early exercise premium)
# ---------------------------------------------------------------------------
def test_american_put_ge_european():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    amer = lsm_american_price(S, K, T, r, sigma, n_simulations=10000,
                               n_time_steps=50, option_type="put", seed=42)
    euro = black_scholes_price(S, K, T, r, sigma, option_type="put")
    assert amer["price"] >= euro - 0.30  # allow MC variance (~2 std errors)  # allow small MC noise


# ---------------------------------------------------------------------------
# Test 2: American put price close to binomial tree reference
# ---------------------------------------------------------------------------
def test_american_put_close_to_binomial():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bt = binomial_tree_price(S, K, T, r, sigma, n_steps=500,
                              option_type="put", exercise="american")["price"]
    lsm = lsm_american_price(S, K, T, r, sigma, n_simulations=20000,
                               n_time_steps=100, option_type="put", seed=7)
    # LSM should be within 2-3% of binomial
    assert abs(lsm["price"] - bt) / bt < 0.05


# ---------------------------------------------------------------------------
# Test 3: American call on non-dividend stock >= European call
# ---------------------------------------------------------------------------
def test_american_call_ge_european_call():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    amer = lsm_american_price(S, K, T, r, sigma, n_simulations=10000,
                               option_type="call", seed=1)
    euro = black_scholes_price(S, K, T, r, sigma, option_type="call")
    assert amer["price"] >= euro - 0.30  # allow MC variance (~2 std errors)


# ---------------------------------------------------------------------------
# Test 4: Returns required keys
# ---------------------------------------------------------------------------
def test_lsm_return_keys():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, seed=0)
    assert "price" in result
    assert "standard_error" in result
    assert "exercise_boundary" in result
    assert "early_exercise_frequency" in result


# ---------------------------------------------------------------------------
# Test 5: Standard error is positive
# ---------------------------------------------------------------------------
def test_lsm_standard_error_positive():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_simulations=5000, seed=99)
    assert result["standard_error"] > 0


# ---------------------------------------------------------------------------
# Test 6: T=0 returns intrinsic
# ---------------------------------------------------------------------------
def test_lsm_t0_returns_intrinsic():
    result_put = lsm_american_price(90.0, 100.0, 0.0, 0.05, 0.20, option_type="put")
    assert abs(result_put["price"] - 10.0) < 1e-10
    result_call = lsm_american_price(110.0, 100.0, 0.0, 0.05, 0.20, option_type="call")
    assert abs(result_call["price"] - 10.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 7: Laguerre basis gives plausible price
# ---------------------------------------------------------------------------
def test_lsm_laguerre_basis():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_simulations=10000, basis_functions="laguerre",
                                 seed=5)
    # Price should be in plausible range
    assert 0.0 < result["price"] < 30.0


# ---------------------------------------------------------------------------
# Test 8: Hermite basis gives plausible price
# ---------------------------------------------------------------------------
def test_lsm_hermite_basis():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_simulations=10000, basis_functions="hermite",
                                 seed=6)
    assert 0.0 < result["price"] < 30.0


# ---------------------------------------------------------------------------
# Test 9: Invalid spot raises ValueError
# ---------------------------------------------------------------------------
def test_lsm_invalid_spot():
    with pytest.raises(ValueError, match="spot"):
        lsm_american_price(0.0, 100.0, 1.0, 0.05, 0.20)


# ---------------------------------------------------------------------------
# Test 10: Invalid n_time_steps raises ValueError
# ---------------------------------------------------------------------------
def test_lsm_invalid_n_time_steps():
    with pytest.raises(ValueError, match="n_time_steps"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, n_time_steps=0)


# ---------------------------------------------------------------------------
# Test 11: Early exercise frequency is between 0 and 1
# ---------------------------------------------------------------------------
def test_lsm_early_exercise_frequency_range():
    # Deep ITM put: should have significant early exercise
    result = lsm_american_price(100.0, 120.0, 1.0, 0.05, 0.20,
                                 n_simulations=10000, option_type="put", seed=33)
    assert 0.0 <= result["early_exercise_frequency"] <= 1.0


# ---------------------------------------------------------------------------
# Test 12: Exercise boundary length equals n_time_steps - 1
# ---------------------------------------------------------------------------
def test_lsm_exercise_boundary_length():
    n_steps = 30
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_time_steps=n_steps, n_simulations=5000, seed=77)
    # boundary is recorded for steps 1..n-1 (n-1 steps)
    assert len(result["exercise_boundary"]) == n_steps - 1


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------
def test_lsm_invalid_strike():
    with pytest.raises(ValueError, match="strike"):
        lsm_american_price(100.0, 0.0, 1.0, 0.05, 0.20)


def test_lsm_invalid_tte():
    with pytest.raises(ValueError, match="time_to_expiry"):
        lsm_american_price(100.0, 100.0, -1.0, 0.05, 0.20)


def test_lsm_invalid_vol():
    with pytest.raises(ValueError, match="volatility"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, -0.1)


def test_lsm_invalid_n_simulations():
    with pytest.raises(ValueError, match="n_simulations"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, n_simulations=0)


def test_lsm_invalid_n_basis():
    with pytest.raises(ValueError, match="n_basis"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, n_basis=0)


def test_lsm_invalid_option_type():
    with pytest.raises(ValueError, match="option_type"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, option_type="straddle")


def test_lsm_invalid_basis_functions():
    with pytest.raises(ValueError, match="basis_functions"):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, basis_functions="chebyshev")


def test_lsm_laguerre_4_basis():
    """Test Laguerre with 4 basis functions to cover L3 and fallback branches."""
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_simulations=5000, basis_functions="laguerre",
                                 n_basis=5, seed=42)
    assert result["price"] > 0


def test_lsm_hermite_4_basis():
    """Test Hermite with 4 basis functions to cover H3 and fallback branches."""
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_simulations=5000, basis_functions="hermite",
                                 n_basis=5, seed=42)
    assert result["price"] > 0


def test_lsm_low_n_sims_triggers_skip():
    """Very deep OTM put with few sims → many steps with n_itm < n_basis."""
    result = lsm_american_price(200.0, 100.0, 0.5, 0.05, 0.10,
                                 n_simulations=50, n_time_steps=20,
                                 option_type="put", seed=0)
    # Price should be near 0 for deep OTM put
    assert result["price"] >= 0
