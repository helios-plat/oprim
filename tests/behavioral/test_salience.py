"""Tests for oprim.behavioral.salience_function and salience_ranking_weights."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.behavioral.salience import salience_function, salience_ranking_weights


# ---------------------------------------------------------------------------
# salience_function tests
# ---------------------------------------------------------------------------


def test_salience_zero_when_equal():
    """sigma(x, x) = 0 for any x."""
    x = np.array([1.0, -2.0, 0.0, 5.0])
    result = salience_function(x, x)
    np.testing.assert_allclose(result, 0.0)


def test_salience_symmetric():
    """sigma(x, y) = sigma(y, x)."""
    x = np.array([1.0, 3.0, -1.0])
    y = np.array([2.0, 1.0, 4.0])
    np.testing.assert_allclose(salience_function(x, y), salience_function(y, x))


def test_salience_output_range():
    """Output is in [0, 1)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-10, 10, 200)
    y = rng.uniform(-10, 10, 200)
    result = salience_function(x, y)
    assert np.all(result >= 0)
    assert np.all(result < 1)


def test_salience_scalar_reference():
    """Scalar reference broadcasts correctly."""
    x = np.array([0.0, 1.0, 2.0])
    result = salience_function(x, 1.0)
    assert result.shape == x.shape
    assert result[1] == pytest.approx(0.0)  # x==reference → 0


def test_salience_theta_must_be_positive():
    """theta <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="theta"):
        salience_function(np.array([1.0]), np.array([2.0]), theta=0.0)


def test_salience_formula():
    """Manual computation matches function output."""
    x = np.array([3.0])
    y = np.array([1.0])
    expected = abs(3.0 - 1.0) / (abs(3.0) + abs(1.0) + 0.1)
    np.testing.assert_allclose(salience_function(x, y), expected)


def test_salience_theta_effect():
    """Larger theta reduces salience magnitude."""
    x = np.array([5.0])
    y = np.array([1.0])
    s_small = salience_function(x, y, theta=0.01)
    s_large = salience_function(x, y, theta=1.0)
    assert s_small > s_large


# ---------------------------------------------------------------------------
# salience_ranking_weights tests
# ---------------------------------------------------------------------------


def test_weights_sum_to_one_1d():
    """Weights sum to 1 for a 1-D array."""
    scores = np.array([0.3, 0.8, 0.1, 0.5])
    weights = salience_ranking_weights(scores, delta=0.7)
    assert weights.sum() == pytest.approx(1.0)


def test_uniform_when_delta_one():
    """delta=1 produces uniform weights regardless of scores."""
    scores = np.array([0.9, 0.1, 0.5, 0.3])
    weights = salience_ranking_weights(scores, delta=1.0)
    n = len(scores)
    np.testing.assert_allclose(weights, np.full(n, 1.0 / n))


def test_highest_salience_gets_highest_weight():
    """The element with highest salience score gets the largest weight for delta<1."""
    scores = np.array([0.1, 0.9, 0.4])
    weights = salience_ranking_weights(scores, delta=0.7)
    assert np.argmax(weights) == 1  # index 1 has highest salience


def test_weights_shape_preserved():
    """Output shape equals input shape."""
    scores = np.ones((3, 4))
    weights = salience_ranking_weights(scores, delta=0.5, rank_dim=1)
    assert weights.shape == (3, 4)


def test_weights_2d_sum_along_rank_dim():
    """Weights sum to 1 along rank_dim for 2-D input."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, (5, 6))
    weights = salience_ranking_weights(scores, delta=0.8, rank_dim=1)
    row_sums = weights.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_invalid_delta_raises():
    """delta <= 0 or delta > 1 raises ValueError."""
    scores = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="delta"):
        salience_ranking_weights(scores, delta=0.0)
    with pytest.raises(ValueError, match="delta"):
        salience_ranking_weights(scores, delta=1.5)


def test_negative_scores_raise():
    """Negative salience scores raise ValueError."""
    scores = np.array([0.5, -0.1])
    with pytest.raises(ValueError, match="salience_scores"):
        salience_ranking_weights(scores)
