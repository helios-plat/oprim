"""Tests for oprim.behavioral.cpt_value_function."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.behavioral.cpt import cpt_value_function


def test_zero_at_reference_default():
    """v(0) = 0 when reference_point=0."""
    assert cpt_value_function(0.0) == pytest.approx(0.0)


def test_unit_gain_alpha_one():
    """v(1) = 1 when alpha=1 and reference_point=0."""
    result = cpt_value_function(1.0, alpha=1.0, beta=1.0, loss_aversion=1.0)
    assert result == pytest.approx(1.0)


def test_unit_loss_default_params():
    """v(-1) = -2.25 at default params (alpha=beta=0.88, lambda=2.25)."""
    # v(-1) = -2.25 * (1)^0.88 = -2.25
    result = cpt_value_function(-1.0)
    assert result == pytest.approx(-2.25)


def test_gain_curvature():
    """Gains are concave: v(2) < 2*v(1) for alpha < 1."""
    v1 = cpt_value_function(1.0)
    v2 = cpt_value_function(2.0)
    assert v2 < 2.0 * v1


def test_loss_convexity():
    """Losses are convex: |v(-2)| < 2*|v(-1)| for beta < 1."""
    v1 = abs(cpt_value_function(-1.0))
    v2 = abs(cpt_value_function(-2.0))
    assert v2 < 2.0 * v1


def test_array_input():
    """Array input returns array output of the same shape."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = cpt_value_function(x)
    assert isinstance(result, np.ndarray)
    assert result.shape == x.shape


def test_scalar_input_returns_float():
    """Scalar input returns a Python float."""
    result = cpt_value_function(1.0)
    assert isinstance(result, float)


def test_reference_point_shift():
    """Using a reference point shifts the kink."""
    # v(5) with ref=5 should be 0
    result = cpt_value_function(5.0, reference_point=5.0)
    assert result == pytest.approx(0.0)
    # v(4) with ref=5 is a loss
    result_loss = cpt_value_function(4.0, reference_point=5.0)
    assert result_loss < 0.0


def test_loss_aversion_scales_losses():
    """Higher loss_aversion yields more negative value for losses."""
    v_low = cpt_value_function(-1.0, loss_aversion=1.5)
    v_high = cpt_value_function(-1.0, loss_aversion=3.0)
    assert v_high < v_low


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        cpt_value_function(1.0, alpha=0.0)


def test_invalid_beta_raises():
    with pytest.raises(ValueError, match="beta"):
        cpt_value_function(-1.0, beta=1.5)


def test_invalid_loss_aversion_raises():
    with pytest.raises(ValueError, match="loss_aversion"):
        cpt_value_function(-1.0, loss_aversion=0.5)


def test_array_values_match_scalar():
    """Vectorized result matches element-wise scalar calls."""
    xs = np.array([-3.0, -1.0, 0.0, 0.5, 2.0])
    arr_result = cpt_value_function(xs)
    scalar_results = np.array([float(cpt_value_function(float(x))) for x in xs])
    np.testing.assert_allclose(arr_result, scalar_results)
