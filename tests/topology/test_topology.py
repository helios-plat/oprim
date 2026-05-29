"""Tests for oprim.topology module."""

import numpy as np
import pytest

from oprim.topology import persistence_landscape, takens_embed


class TestTakensEmbed:
    def test_shape(self):
        """x=range(10), d=3, tau=1 -> shape (8, 3)."""
        x = np.arange(10, dtype=float)
        result = takens_embed(x, d=3, tau=1)
        assert result.shape == (8, 3)

    def test_tau_2(self):
        """tau=2 changes output rows: n = len(x) - (d-1)*tau."""
        x = np.arange(20, dtype=float)
        result = takens_embed(x, d=3, tau=2)
        expected_rows = len(x) - (3 - 1) * 2  # 20 - 4 = 16
        assert result.shape == (expected_rows, 3)

    def test_too_short_raises(self):
        """x too short for d and tau raises ValueError."""
        x = np.array([1.0, 2.0])  # n = 2 - (5-1)*1 = -2 <= 0
        with pytest.raises(ValueError, match="too short"):
            takens_embed(x, d=5, tau=1)

    def test_preserves_values(self):
        """First row is x[0], x[tau], x[2*tau] ..."""
        x = np.arange(10, dtype=float)
        result = takens_embed(x, d=3, tau=1)
        np.testing.assert_array_equal(result[0], [0.0, 1.0, 2.0])

    def test_preserves_values_tau2(self):
        """With tau=2, first row is x[0], x[2], x[4]."""
        x = np.arange(10, dtype=float)
        result = takens_embed(x, d=3, tau=2)
        np.testing.assert_array_equal(result[0], [0.0, 2.0, 4.0])

    def test_d1_trivial(self):
        """d=1 returns a column vector of all elements."""
        x = np.array([1.0, 2.0, 3.0])
        result = takens_embed(x, d=1, tau=1)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result[:, 0], x)


class TestPersistenceLandscape:
    def test_empty_dgm(self):
        """Empty diagram -> zeros(resolution)."""
        result = persistence_landscape(np.empty((0, 2)), resolution=100)
        assert result.shape == (100,)
        assert np.all(result == 0.0)

    def test_single_point(self):
        """Single (birth, death) pair: verify tent function peak."""
        b, d = 0.5, 1.5
        dgm = np.array([[b, d]])
        resolution = 200
        result = persistence_landscape(dgm, resolution=resolution, x_min=0.0, x_max=2.0)
        # Mid-point t = (b+d)/2 = 1.0; tent value = (d-b)/2 = 0.5
        t_vals = np.linspace(0.0, 2.0, resolution)
        mid_idx = np.argmin(np.abs(t_vals - 1.0))
        assert result[mid_idx] == pytest.approx(0.5, abs=0.05)

    def test_k_gt_len(self):
        """k > len(dgm) -> zeros."""
        dgm = np.array([[0.0, 1.0]])
        result = persistence_landscape(dgm, k=5, resolution=50)
        assert np.all(result == 0.0)

    def test_infinite_death(self):
        """d=inf -> clamped to x_max."""
        dgm = np.array([[0.0, np.inf]])
        result = persistence_landscape(dgm, resolution=100, x_min=0.0, x_max=2.0)
        # Should be computed without error and return finite values
        assert np.all(np.isfinite(result))

    def test_resolution_shape(self):
        """Output shape matches resolution parameter."""
        dgm = np.array([[0.0, 1.0], [0.5, 1.5]])
        for res in [50, 100, 200]:
            result = persistence_landscape(dgm, resolution=res)
            assert result.shape == (res,)

    @pytest.mark.academic_reference
    def test_persistence_landscape_bubenik_2015(self):
        """Bubenik (2015): tent(t) = max(0, min(t-birth, death-t)).

        At midpoint t = (b+d)/2, tent value = (d-b)/2.
        """
        b, d = 0.0, 2.0
        dgm = np.array([[b, d]])
        resolution = 1000
        result = persistence_landscape(dgm, resolution=resolution, x_min=0.0, x_max=2.0)
        t_vals = np.linspace(0.0, 2.0, resolution)
        # At midpoint t=1.0: expected tent = (2.0 - 0.0) / 2 = 1.0
        mid_idx = np.argmin(np.abs(t_vals - 1.0))
        assert result[mid_idx] == pytest.approx(1.0, abs=0.02)

    def test_multiple_points_k1(self):
        """k=1 gives largest tent at each t."""
        dgm = np.array([[0.0, 2.0], [0.5, 1.5]])
        result = persistence_landscape(dgm, k=1, resolution=100, x_min=0.0, x_max=2.0)
        # All values must be >= 0
        assert np.all(result >= 0)
