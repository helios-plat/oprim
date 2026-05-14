"""Tests for svensson_yield_curve and cubic_spline_yield_curve."""
from __future__ import annotations

import pytest
import numpy as np
from scipy.interpolate import CubicSpline

from oprim.derivatives.rates import svensson_yield_curve, cubic_spline_yield_curve


# ===========================================================================
# Helpers: synthetic yield curves
# ===========================================================================

def _synthetic_svensson_yields(maturities, params):
    """Generate synthetic yields from known Svensson params."""
    from oprim.derivatives.rates import _svensson_yield
    return _svensson_yield(np.asarray(maturities), params)


_MATURITIES = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0])

_TRUE_PARAMS = {
    "beta_0": 0.06,
    "beta_1": -0.03,
    "beta_2": 0.04,
    "beta_3": 0.02,
    "tau_1": 1.5,
    "tau_2": 6.0,
}


# ===========================================================================
# svensson_yield_curve tests (≥7)
# ===========================================================================

# Test 1: Fits clean synthetic data with RMSE < 0.001
def test_svensson_fits_clean_data():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    assert result["rmse"] < 0.001


# Test 2: Returns correct keys
def test_svensson_return_keys():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    assert "params" in result
    assert "fitted_yields" in result
    assert "residuals" in result
    assert "rmse" in result
    assert "converged" in result


# Test 3: Fitted yields shape matches input
def test_svensson_fitted_yields_shape():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    assert len(result["fitted_yields"]) == len(_MATURITIES)


# Test 4: Params dict has correct keys
def test_svensson_params_keys():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    expected_keys = {"beta_0", "beta_1", "beta_2", "beta_3", "tau_1", "tau_2"}
    assert set(result["params"].keys()) == expected_keys


# Test 5: Residuals are near zero on clean data
def test_svensson_residuals_near_zero():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    assert np.all(np.abs(result["residuals"]) < 0.01)


# Test 6: Works with custom initial params
def test_svensson_custom_initial_params():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    custom_init = {"beta_0": 0.07, "tau_1": 2.0, "tau_2": 7.0}
    result = svensson_yield_curve(_MATURITIES, y, initial_params=custom_init)
    assert result["rmse"] < 0.002


# Test 7: Invalid inputs raise ValueError
def test_svensson_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        svensson_yield_curve(np.array([1.0, 2.0]), np.array([0.05]))


def test_svensson_negative_maturity():
    with pytest.raises(ValueError, match="maturities must be"):
        svensson_yield_curve(np.array([-1.0, 2.0]), np.array([0.05, 0.06]))


# Test 8: tau_1, tau_2 remain positive after fitting
def test_svensson_tau_positive():
    y = _synthetic_svensson_yields(_MATURITIES, _TRUE_PARAMS)
    result = svensson_yield_curve(_MATURITIES, y)
    assert result["params"]["tau_1"] > 0
    assert result["params"]["tau_2"] > 0


# ===========================================================================
# cubic_spline_yield_curve tests (≥6)
# ===========================================================================

# Test 9: Interpolates exactly at knots
def test_cubic_spline_exact_at_knots():
    t = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    y = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    result = cubic_spline_yield_curve(t, y)
    fitted = result["fitted_yields"]
    np.testing.assert_allclose(fitted, y, atol=1e-10)


# Test 10: Returns correct keys
def test_cubic_spline_return_keys():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    y = np.array([0.02, 0.025, 0.03, 0.035])
    result = cubic_spline_yield_curve(t, y)
    assert "spline_object" in result
    assert "fitted_yields" in result
    assert "derivative_coefficients" in result
    assert "evaluate" in result


# Test 11: spline_object is a CubicSpline
def test_cubic_spline_object_type():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    y = np.array([0.02, 0.025, 0.03, 0.035])
    result = cubic_spline_yield_curve(t, y)
    assert isinstance(result["spline_object"], CubicSpline)


# Test 12: evaluate callable works on new points
def test_cubic_spline_evaluate_callable():
    t = np.array([1.0, 2.0, 5.0, 10.0])
    y = np.array([0.02, 0.025, 0.03, 0.035])
    result = cubic_spline_yield_curve(t, y)
    new_t = np.array([1.5, 3.0, 7.5])
    vals = result["evaluate"](new_t)
    assert len(vals) == 3
    # Values should be interpolated in reasonable range
    assert np.all(vals >= 0.01)
    assert np.all(vals <= 0.05)


# Test 13: Different boundary conditions give different spline shapes between knots
def test_cubic_spline_different_boundary():
    t = np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    y = np.array([0.02, 0.025, 0.03, 0.04, 0.035])  # non-monotone to differentiate
    res_nat = cubic_spline_yield_curve(t, y, boundary_type="natural")
    res_nak = cubic_spline_yield_curve(t, y, boundary_type="not_a_knot")
    # Evaluate at midpoints between knots — boundary conditions affect shape between nodes
    mid = np.array([1.5, 3.5, 7.5, 15.0])
    val_nat = res_nat["evaluate"](mid)
    val_nak = res_nak["evaluate"](mid)
    assert not np.allclose(val_nat, val_nak)


# Test 14: Mismatched lengths raises ValueError
def test_cubic_spline_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cubic_spline_yield_curve(np.array([1.0, 2.0]), np.array([0.02]))


# Test 15: Fewer than 2 points raises ValueError
def test_cubic_spline_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        cubic_spline_yield_curve(np.array([1.0]), np.array([0.02]))


# Test 16: Sorted input handles out-of-order maturities
def test_cubic_spline_unsorted_input():
    t = np.array([5.0, 1.0, 10.0, 2.0])
    y = np.array([0.03, 0.02, 0.035, 0.025])
    result = cubic_spline_yield_curve(t, y)
    # Should succeed and evaluate correctly at a midpoint
    val = result["evaluate"](3.0)
    assert 0.01 < float(val) < 0.05
