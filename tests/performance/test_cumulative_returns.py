"""Tests for oprim.performance.cumulative_returns."""
import math
import numpy as np
import pandas as pd
import pytest

from oprim.performance.cumulative import cumulative_returns


class TestCumulativeReturnsSimple:
    def test_cumulative_returns_simple_basic(self):
        """[0.1, 0.1] simple → [1.1, 1.21]."""
        result = cumulative_returns([0.1, 0.1], method="simple")
        np.testing.assert_allclose(result, [1.1, 1.21], rtol=1e-12)

    def test_cumulative_returns_log_basic(self):
        """[0.1, 0.1] log → [exp(0.1), exp(0.2)] * starting_value."""
        result = cumulative_returns([0.1, 0.1], method="log")
        expected = [math.exp(0.1), math.exp(0.2)]
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_cumulative_returns_zero_input(self):
        """Zeros → flat line at starting_value."""
        result = cumulative_returns([0.0, 0.0, 0.0], starting_value=100.0, method="simple")
        np.testing.assert_allclose(result, [100.0, 100.0, 100.0], rtol=1e-12)

    def test_cumulative_returns_negative_returns_compound(self):
        """Negative returns should decay correctly."""
        result = cumulative_returns([-0.1, -0.1], method="simple")
        expected = [0.9, 0.81]
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_cumulative_returns_starting_value_default_1(self):
        """Default starting_value=1.0."""
        result = cumulative_returns([0.05])
        assert result[0] == pytest.approx(1.05, rel=1e-12)

    def test_cumulative_returns_invalid_method_raises(self):
        """Unknown method → ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            cumulative_returns([0.1], method="compound")

    def test_cumulative_returns_preserves_series_index(self):
        """pd.Series input preserves its index."""
        idx = pd.date_range("2020-01-01", periods=3)
        s = pd.Series([0.01, 0.02, -0.01], index=idx)
        result = cumulative_returns(s)
        assert isinstance(result, pd.Series)
        assert list(result.index) == list(idx)

    @pytest.mark.academic_reference
    def test_cumulative_returns_matches_known_formula(self):
        """[0.05]*4 simple → 1.0*1.05^4=1.21550625, rtol=1e-12."""
        result = cumulative_returns([0.05, 0.05, 0.05, 0.05], method="simple")
        expected_final = 1.0 * 1.05**4
        np.testing.assert_allclose(result[-1], expected_final, rtol=1e-12)


class TestCumulativeReturnsEdgeCases:
    def test_cumulative_returns_invalid_starting_value_raises(self):
        """starting_value=0 → ValueError."""
        with pytest.raises(ValueError):
            cumulative_returns([0.1], starting_value=0.0)

    def test_cumulative_returns_negative_starting_value_raises(self):
        """starting_value=-1 → ValueError."""
        with pytest.raises(ValueError):
            cumulative_returns([0.1], starting_value=-1.0)

    def test_cumulative_returns_log_zero_input(self):
        """Log method with zero returns → flat line at starting_value."""
        result = cumulative_returns([0.0, 0.0, 0.0], method="log", starting_value=2.0)
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0], rtol=1e-12)

    def test_cumulative_returns_ndarray_output_type(self):
        """ndarray input → ndarray output."""
        arr = np.array([0.1, 0.2])
        result = cumulative_returns(arr)
        assert isinstance(result, np.ndarray)
