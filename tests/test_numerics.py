"""Tests for oprim.numerics module."""

import logging
import warnings

import numpy as np
import pytest
from scipy.special import logsumexp as scipy_logsumexp
from scipy.special import softmax as scipy_softmax

from oprim.numerics import clip_with_warning, logsumexp_safe, softmax_safe


# ============================================================
# logsumexp_safe
# ============================================================
class TestLogsumexpSafe:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = logsumexp_safe(x)
        expected = scipy_logsumexp(x)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_overflow_prevention(self):
        """Large values should not overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = logsumexp_safe(x)
        expected = scipy_logsumexp(x)
        np.testing.assert_allclose(result, expected, rtol=1e-9)
        assert np.isfinite(result)

    def test_underflow_prevention(self):
        """Very small values should not underflow to -inf."""
        x = np.array([-1000.0, -999.0, -998.0])
        result = logsumexp_safe(x)
        assert np.isfinite(result)

    def test_all_neg_inf(self):
        x = np.array([-np.inf, -np.inf, -np.inf])
        result = logsumexp_safe(x)
        assert result == -np.inf

    def test_with_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0, 2.0, 3.0])
        result = logsumexp_safe(x, weights=w)
        expected = scipy_logsumexp(x, b=w)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_axis(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp_safe(x, axis=1)
        expected = scipy_logsumexp(x, axis=1)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_keepdims(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = logsumexp_safe(x, axis=1, keepdims=True)
        assert result.shape == (2, 1)

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 10, (50, 20))
        result = logsumexp_safe(x, axis=1)
        expected = scipy_logsumexp(x, axis=1)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# softmax_safe
# ============================================================
class TestSoftmaxSafe:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax_safe(x)
        assert np.sum(result) == pytest.approx(1.0, abs=1e-9)

    def test_high_temperature_uniform(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = softmax_safe(x, temperature=1000.0)
        # Should be approximately uniform
        np.testing.assert_allclose(result, np.ones(5) / 5, atol=0.01)

    def test_low_temperature_one_hot(self):
        x = np.array([1.0, 2.0, 10.0, 3.0])
        result = softmax_safe(x, temperature=0.01)
        # Should be approximately one-hot at index 2
        assert result[2] > 0.99

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            softmax_safe(np.array([1.0, 2.0]), temperature=0)

    def test_2d_axis(self):
        x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        result = softmax_safe(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), [1.0, 1.0], rtol=1e-9)

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 5, 20)
        result = softmax_safe(x, temperature=1.0)
        expected = scipy_softmax(x)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# clip_with_warning
# ============================================================
class TestClipWithWarning:
    def test_no_clip(self):
        x = np.array([0.1, 0.2, 0.3])
        result = clip_with_warning(x, lower=0.0, upper=1.0)
        np.testing.assert_array_equal(result, x)

    def test_clip_below_threshold(self):
        """1% clipped, below 5% threshold → no warning."""
        x = np.arange(100.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clip_with_warning(x, upper=98.5)
            assert len(w) == 0

    def test_clip_above_threshold(self):
        """20% clipped → warning."""
        x = np.arange(100.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clip_with_warning(x, upper=79.0)
            assert len(w) == 1
            assert "clipped" in str(w[0].message)

    def test_scalar_input(self):
        result = clip_with_warning(5.0, upper=3.0)
        assert result == 3.0
        assert isinstance(result, float)

    def test_logger(self):
        logger = logging.getLogger("test_clip")
        handler = logging.handlers = []
        x = np.zeros(10)
        x[0:8] = 100  # 80% will be clipped
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            clip_with_warning(x, upper=50.0, logger=logger)
            # When logger is provided, no warnings module warning
            assert len(w) == 0

    def test_lower_only(self):
        x = np.array([-5.0, 0.0, 5.0])
        result = clip_with_warning(x, lower=0.0)
        assert result[0] == 0.0
        assert result[2] == 5.0

    def test_upper_only(self):
        x = np.array([0.0, 5.0, 10.0])
        result = clip_with_warning(x, upper=7.0)
        assert result[2] == 7.0
