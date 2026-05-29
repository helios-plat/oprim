"""Tests for oprim.behavioral.probability_weighting_function."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from oprim.behavioral.weighting import probability_weighting_function


def test_tk_boundary_zero():
    """w(0) = 0 for TK form."""
    assert probability_weighting_function(0.0, form="tk") == pytest.approx(0.0)


def test_tk_boundary_one():
    """w(1) = 1 for TK form."""
    assert probability_weighting_function(1.0, form="tk") == pytest.approx(1.0)


def test_prelec_boundary_zero():
    """w(0) = 0 for Prelec form."""
    assert probability_weighting_function(0.0, form="prelec") == pytest.approx(0.0)


def test_prelec_boundary_one():
    """w(1) = 1 for Prelec form."""
    assert probability_weighting_function(1.0, form="prelec") == pytest.approx(1.0)


def test_prelec_fixed_point():
    """Prelec: w(1/e) = exp(-1) when delta=1, gamma=1."""
    p = np.exp(-1.0)
    result = probability_weighting_function(p, form="prelec", gamma_gain=1.0, delta=1.0)
    assert result == pytest.approx(np.exp(-1.0), rel=1e-9)


def test_tk_gain_overweights_small_probs():
    """TK inverse-S shape: w(p) > p for small p at default gamma=0.61."""
    # At gamma=0.61, small probs are overweighted; e.g. w(0.1) > 0.1
    result = probability_weighting_function(0.1, form="tk", side="gain")
    assert result > 0.1


def test_tk_array_input():
    """Array input returns same-shape array for TK."""
    p_arr = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    result = probability_weighting_function(p_arr, form="tk")
    assert isinstance(result, np.ndarray)
    assert result.shape == p_arr.shape
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_prelec_array_input():
    """Array input returns same-shape array for Prelec."""
    p_arr = np.linspace(0, 1, 11)
    result = probability_weighting_function(p_arr, form="prelec")
    assert isinstance(result, np.ndarray)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


def test_output_in_unit_interval():
    """All output values lie in [0, 1] for interior probabilities."""
    p_arr = np.linspace(0.01, 0.99, 50)
    result = probability_weighting_function(p_arr, form="tk")
    assert np.all(result >= 0)
    assert np.all(result <= 1)


def test_scalar_returns_float():
    """Scalar input returns Python float."""
    result = probability_weighting_function(0.5)
    assert isinstance(result, float)


def test_loss_side_uses_gamma_loss():
    """Loss side uses gamma_loss, so result differs from gain side."""
    gain = probability_weighting_function(0.3, form="tk", side="gain")
    loss = probability_weighting_function(0.3, form="tk", side="loss")
    # gamma_gain=0.61 != gamma_loss=0.69 → different values
    assert gain != pytest.approx(loss)


def test_invalid_p_raises():
    with pytest.raises(ValueError, match="p must be in"):
        probability_weighting_function(1.5, form="tk")


def test_invalid_gamma_raises():
    with pytest.raises(ValueError, match="gamma_gain"):
        probability_weighting_function(0.5, gamma_gain=0.0)


def test_small_gamma_warns():
    """TK with gamma < 0.28 should emit a UserWarning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probability_weighting_function(0.5, form="tk", gamma_gain=0.2)
    assert any(issubclass(w.category, UserWarning) for w in caught)
