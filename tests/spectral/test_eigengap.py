"""Tests for oprim.spectral.spectral_eigengap_detect."""
from __future__ import annotations

import numpy as np
import pytest

from oprim.spectral.eigengap import spectral_eigengap_detect


class TestSpectralEigengapDetect:
    def test_returns_all_keys(self):
        eigs = np.array([10.0, 5.0, 1.0, 0.9, 0.8])
        result = spectral_eigengap_detect(eigs)
        assert set(result.keys()) == {"k_star", "gaps", "confidence"}

    def test_two_cluster_structure_largest_gap(self):
        # Clear gap after 2nd eigenvalue → k_star=2
        eigs = np.array([10.0, 9.0, 1.0, 0.9, 0.8, 0.7])
        result = spectral_eigengap_detect(eigs, method="largest_gap")
        assert result["k_star"] == 2

    def test_single_dominant_eigenvalue(self):
        # Huge first eigenvalue, rest small → k_star=1
        eigs = np.array([100.0, 1.0, 0.9, 0.8])
        result = spectral_eigengap_detect(eigs, method="largest_gap")
        assert result["k_star"] == 1

    def test_relative_method_two_clusters(self):
        eigs = np.array([10.0, 9.5, 1.0, 0.95, 0.9])
        result = spectral_eigengap_detect(eigs, method="relative")
        assert result["k_star"] == 2

    def test_elbow_method_returns_valid_k(self):
        eigs = np.array([10.0, 8.0, 4.0, 1.0, 0.5, 0.4])
        result = spectral_eigengap_detect(eigs, method="elbow")
        assert 1 <= result["k_star"] <= len(eigs) - 1

    def test_max_k_limits_search(self):
        # With max_k=1 the answer must be 1
        eigs = np.array([10.0, 9.0, 1.0, 0.5])
        result = spectral_eigengap_detect(eigs, method="largest_gap", max_k=1)
        assert result["k_star"] == 1

    def test_unsorted_input_handled(self):
        # Unsorted order should give same result as sorted
        eigs_sorted = np.array([10.0, 9.0, 1.0, 0.8])
        eigs_random = np.array([1.0, 9.0, 10.0, 0.8])
        r1 = spectral_eigengap_detect(eigs_sorted, method="largest_gap")
        r2 = spectral_eigengap_detect(eigs_random, method="largest_gap")
        assert r1["k_star"] == r2["k_star"]

    def test_confidence_is_float(self):
        eigs = np.array([5.0, 4.0, 1.0, 0.5])
        result = spectral_eigengap_detect(eigs)
        assert isinstance(result["confidence"], float)

    def test_confidence_gte_one_for_dominant_gap(self):
        # The chosen gap is the largest, confidence >= 1
        eigs = np.array([100.0, 1.0, 0.9, 0.8])
        result = spectral_eigengap_detect(eigs, method="largest_gap")
        assert result["confidence"] >= 1.0

    def test_too_few_eigenvalues_raises(self):
        with pytest.raises(ValueError):
            spectral_eigengap_detect(np.array([1.0]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            spectral_eigengap_detect(np.array([3.0, 2.0, 1.0]), method="bogus")  # type: ignore[arg-type]

    def test_gaps_shape(self):
        eigs = np.array([5.0, 4.0, 3.0, 1.0])
        result = spectral_eigengap_detect(eigs, method="largest_gap")
        # largest_gap gaps: n-1 elements
        assert len(result["gaps"]) == len(eigs) - 1
