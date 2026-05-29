"""Tests for oprim.spectral.rotationally_invariant_estimator."""
from __future__ import annotations

import numpy as np
import pytest

from oprim.spectral.rie import rotationally_invariant_estimator


def _make_identity_cov(n: int) -> np.ndarray:
    return np.eye(n)


def _make_random_cov(n: int, t: int, seed: int = 42) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, n))
    cov = np.cov(x.T, ddof=1)
    return cov, t


class TestRIEBouchaud:
    def test_returns_all_keys(self):
        cov, t = _make_random_cov(10, 100)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert set(result.keys()) == {
            "cov_rie", "eigenvalues_raw", "eigenvalues_clean", "stieltjes_estimate"
        }

    def test_output_is_symmetric(self):
        cov, t = _make_random_cov(10, 100)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert np.allclose(result["cov_rie"], result["cov_rie"].T, atol=1e-10)

    def test_output_is_psd(self):
        cov, t = _make_random_cov(10, 100)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        eigs = np.linalg.eigvalsh(result["cov_rie"])
        assert np.all(eigs >= -1e-9)

    def test_identity_input_structure(self):
        # Identity cov → RIE preserves diagonal structure (off-diagonals remain 0)
        cov = _make_identity_cov(5)
        result = rotationally_invariant_estimator(cov, n_samples=10000)
        cov_rie = result["cov_rie"]
        off_diag = cov_rie - np.diag(np.diag(cov_rie))
        assert np.allclose(off_diag, 0.0, atol=1e-10)
        # All cleaned eigenvalues should be equal (symmetric degenerate case)
        xi = result["eigenvalues_clean"]
        assert np.allclose(xi, xi[0], rtol=1e-8)

    def test_eigenvalues_clean_shape_matches(self):
        cov, t = _make_random_cov(8, 200)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert result["eigenvalues_raw"].shape == result["eigenvalues_clean"].shape

    def test_bouchaud_shrinkage_reduces_top_eigenvalue(self):
        # Bouchaud RIE should not inflate eigenvalues vs sample
        cov, t = _make_random_cov(15, 50)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert result["eigenvalues_clean"].max() <= result["eigenvalues_raw"].max() + 1e-6

    def test_n_features_gt_n_samples_handled(self):
        # Wide matrix: n_features > n_samples — should not raise
        cov, _ = _make_random_cov(20, 15)
        result = rotationally_invariant_estimator(cov, n_samples=15)
        assert result["cov_rie"].shape == (20, 20)

    def test_stieltjes_estimate_shape(self):
        cov, t = _make_random_cov(6, 100)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert result["stieltjes_estimate"].shape == (6,)

    def test_eigenvalues_clean_positive(self):
        cov, t = _make_random_cov(10, 30)
        result = rotationally_invariant_estimator(cov, n_samples=t)
        assert np.all(result["eigenvalues_clean"] > 0)

    def test_invalid_non_square_raises(self):
        with pytest.raises(ValueError):
            rotationally_invariant_estimator(np.ones((3, 4)), n_samples=10)

    def test_invalid_n_samples_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            rotationally_invariant_estimator(np.eye(4), n_samples=0)

    def test_ledoit_peche_method(self):
        cov, t = _make_random_cov(8, 100)
        result = rotationally_invariant_estimator(cov, n_samples=t, method="ledoit_peche")
        assert result["cov_rie"].shape == (8, 8)
        eigs = np.linalg.eigvalsh(result["cov_rie"])
        assert np.all(eigs >= -1e-9)

    def test_unknown_method_raises(self):
        cov, t = _make_random_cov(5, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            rotationally_invariant_estimator(cov, n_samples=t, method="invalid")  # type: ignore[arg-type]
