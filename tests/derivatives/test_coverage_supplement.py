"""Supplemental tests for coverage of edge cases in Phase 5A derivatives."""
from __future__ import annotations

import numpy as np
import pytest

from oprim.derivatives.american import lsm_american_price
from oprim.derivatives.binomial_tree import binomial_tree_price
from oprim.derivatives.exotic import barrier_option_price, lookback_option_price
from oprim.derivatives.monte_carlo import mc_asian_price, mc_european_price
from oprim.derivatives.rates import cubic_spline_yield_curve, svensson_yield_curve


# ---------------------------------------------------------------------------
# american.py – laguerre / hermite basis and validation edges
# ---------------------------------------------------------------------------
def test_lsm_laguerre_basis():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 basis_functions="laguerre", seed=0)
    assert result["price"] > 0


def test_lsm_hermite_basis():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 basis_functions="hermite", seed=0)
    assert result["price"] > 0


def test_lsm_invalid_spot():
    with pytest.raises(ValueError):
        lsm_american_price(0.0, 100.0, 1.0, 0.05, 0.20)


def test_lsm_invalid_n_time_steps():
    with pytest.raises(ValueError):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, n_time_steps=0)


def test_lsm_invalid_basis():
    with pytest.raises(ValueError):
        lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20, basis_functions="fourier")


def test_lsm_t_zero_call():
    result = lsm_american_price(110.0, 100.0, 0.0, 0.05, 0.20, option_type="call")
    assert result["price"] == pytest.approx(10.0, abs=0.01)


def test_lsm_t_zero_put():
    result = lsm_american_price(90.0, 100.0, 0.0, 0.05, 0.20, option_type="put")
    assert result["price"] == pytest.approx(10.0, abs=0.01)


def test_lsm_n_basis_4():
    result = lsm_american_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                 n_basis=4, seed=0)
    assert result["price"] > 0


# ---------------------------------------------------------------------------
# exotic.py – up-and-in / down-and-in barriers, sigma=0 edges
# ---------------------------------------------------------------------------
def test_barrier_up_and_in_call():
    result = barrier_option_price(100.0, 100.0, 110.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_in", option_type="call")
    assert result["price"] >= 0


def test_barrier_down_and_in_put():
    result = barrier_option_price(100.0, 100.0, 90.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_in", option_type="put")
    assert result["price"] >= 0


def test_barrier_up_and_out_put():
    result = barrier_option_price(100.0, 100.0, 120.0, 1.0, 0.05, 0.20,
                                   barrier_type="up_and_out", option_type="put")
    assert result["price"] >= 0


def test_barrier_down_and_out_call():
    result = barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_out", option_type="call")
    assert result["price"] >= 0


def test_barrier_with_rebate():
    result = barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                                   barrier_type="down_and_out", rebate=5.0)
    assert result["price"] >= 0


def test_barrier_t_zero():
    result = barrier_option_price(100.0, 90.0, 80.0, 0.0, 0.05, 0.20,
                                   barrier_type="down_and_out", option_type="call")
    assert result["price"] == pytest.approx(10.0, abs=0.01)


def test_barrier_invalid_barrier_type():
    with pytest.raises(ValueError):
        barrier_option_price(100.0, 100.0, 80.0, 1.0, 0.05, 0.20,
                             barrier_type="sideways_and_confused")


def test_lookback_fixed_call():
    result = lookback_option_price(100.0, 95.0, 1.0, 0.05, 0.20,
                                    strike_type="fixed", option_type="call")
    assert result["price"] > 0


def test_lookback_fixed_put():
    result = lookback_option_price(100.0, 105.0, 1.0, 0.05, 0.20,
                                    strike_type="fixed", option_type="put")
    assert result["price"] > 0


def test_lookback_mc():
    result = lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                    method="monte_carlo", n_simulations=2000)
    assert result["price"] > 0


def test_lookback_invalid_method():
    with pytest.raises(ValueError):
        lookback_option_price(100.0, 100.0, 1.0, 0.05, 0.20, method="tarot_cards")


# ---------------------------------------------------------------------------
# monte_carlo.py – control_variate, sigma=0, asian MC
# ---------------------------------------------------------------------------
def test_mc_european_control_variate():
    result = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                n_simulations=5000, antithetic=False,
                                control_variate=True, seed=0)
    assert result["price"] > 0
    assert "control_variate" in result["method"]


def test_mc_european_no_variance_reduction():
    result = mc_european_price(100.0, 100.0, 1.0, 0.05, 0.20,
                                n_simulations=5000, antithetic=False,
                                control_variate=False, seed=0)
    assert result["method"] == "monte_carlo"


def test_mc_european_sigma_zero():
    # sigma=0: deterministic → BS-like formula
    result = mc_european_price(110.0, 100.0, 1.0, 0.05, 0.0,
                                n_simulations=500, seed=0)
    assert result["price"] > 0


def test_mc_asian_geometric():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             averaging="geometric", n_simulations=2000, seed=0)
    assert result["price"] > 0


def test_mc_asian_floating():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             strike_type="floating", n_simulations=2000, seed=0)
    assert result["price"] > 0


def test_mc_asian_put():
    result = mc_asian_price(100.0, 100.0, 1.0, 0.05, 0.20,
                             option_type="put", n_simulations=2000, seed=0)
    assert result["price"] > 0


# ---------------------------------------------------------------------------
# rates.py – validation errors, error paths
# ---------------------------------------------------------------------------
def test_svensson_mismatched_lengths():
    with pytest.raises(ValueError):
        svensson_yield_curve(np.array([1.0, 2.0]), np.array([0.02]))


def test_svensson_too_few_points():
    with pytest.raises(ValueError):
        svensson_yield_curve(np.array([1.0]), np.array([0.02]))


def test_svensson_negative_tau_initial():
    # Provide initial params with valid bounds; should still converge
    t = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    y = np.array([0.015, 0.018, 0.022, 0.028, 0.035, 0.038, 0.040])
    result = svensson_yield_curve(t, y,
                                   initial_params={"beta_0": 0.04, "beta_1": -0.02,
                                                   "beta_2": 0.01, "beta_3": 0.01,
                                                   "tau_1": 1.0, "tau_2": 3.0})
    assert result["rmse"] < 0.005


def test_cubic_spline_clamped():
    t = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    y = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    result = cubic_spline_yield_curve(t, y, boundary_type="clamped")
    assert np.allclose(result["fitted_yields"], y, atol=1e-10)


def test_cubic_spline_single_point_raises():
    with pytest.raises(ValueError):
        cubic_spline_yield_curve(np.array([1.0]), np.array([0.02]))
