"""Tests for fisher_rao_distance."""

from __future__ import annotations

import math

import numpy as np
import pytest

from oprim.info_geometry.fisher_rao import fisher_rao_distance


def test_fisher_rao_gaussian_same_distribution():
    a = {"mean": 0.0, "std": 1.0}
    r = fisher_rao_distance(a, a, distribution_family="univariate_gaussian")
    assert abs(r["distance"]) < 1e-10


def test_fisher_rao_gaussian_different_means():
    a = {"mean": 0.0, "std": 1.0}
    b = {"mean": 1.0, "std": 1.0}
    r = fisher_rao_distance(a, b, distribution_family="univariate_gaussian")
    assert r["distance"] > 0


def test_fisher_rao_gaussian_symmetric():
    a = {"mean": 0.0, "std": 1.0}
    b = {"mean": 2.0, "std": 0.5}
    r1 = fisher_rao_distance(a, b, distribution_family="univariate_gaussian")
    r2 = fisher_rao_distance(b, a, distribution_family="univariate_gaussian")
    np.testing.assert_allclose(r1["distance"], r2["distance"], rtol=1e-10)


def test_fisher_rao_gaussian_closed_form():
    # Check closed-form result manually
    mu1, sigma1 = 0.0, 1.0
    mu2, sigma2 = 0.0, 2.0
    expected = math.sqrt(2) * math.acosh(
        1 + ((mu1 - mu2) ** 2 + 2 * (sigma1 - sigma2) ** 2) / (4 * sigma1 * sigma2)
    )
    a = {"mean": mu1, "std": sigma1}
    b = {"mean": mu2, "std": sigma2}
    r = fisher_rao_distance(a, b, distribution_family="univariate_gaussian")
    np.testing.assert_allclose(r["distance"], expected, rtol=1e-8)


def test_fisher_rao_categorical_same():
    a = {"probs": np.array([0.5, 0.5])}
    r = fisher_rao_distance(a, a, distribution_family="categorical")
    assert abs(r["distance"]) < 1e-10


def test_fisher_rao_categorical_opposite():
    a = {"probs": np.array([1.0, 0.0])}
    b = {"probs": np.array([0.0, 1.0])}
    r = fisher_rao_distance(a, b, distribution_family="categorical")
    # arccos(0) = pi/2, times 2 = pi
    np.testing.assert_allclose(r["distance"], np.pi, rtol=1e-6)


def test_fisher_rao_categorical_formula():
    # Bhattacharyya angle
    a = {"probs": np.array([0.5, 0.5])}
    b = {"probs": np.array([0.25, 0.75])}
    bc = np.sum(np.sqrt(a["probs"] * b["probs"]))
    expected = 2 * np.arccos(bc)
    r = fisher_rao_distance(a, b, distribution_family="categorical")
    np.testing.assert_allclose(r["distance"], expected, rtol=1e-8)


def test_fisher_rao_multivariate_same():
    a = {"mean": np.zeros(2), "cov": np.eye(2)}
    r = fisher_rao_distance(a, a, distribution_family="multivariate_gaussian")
    assert abs(r["distance"]) < 1e-10


def test_fisher_rao_multivariate_positive():
    a = {"mean": np.zeros(2), "cov": np.eye(2)}
    b = {"mean": np.array([1.0, 0.0]), "cov": np.eye(2)}
    r = fisher_rao_distance(a, b, distribution_family="multivariate_gaussian")
    assert r["distance"] > 0


def test_fisher_rao_invalid_std_raises():
    a = {"mean": 0.0, "std": -1.0}
    b = {"mean": 0.0, "std": 1.0}
    with pytest.raises(ValueError):
        fisher_rao_distance(a, b, distribution_family="univariate_gaussian")


def test_fisher_rao_invalid_probs_raises():
    a = {"probs": np.array([0.5, 0.3])}  # doesn't sum to 1
    b = {"probs": np.array([0.5, 0.5])}
    with pytest.raises(ValueError):
        fisher_rao_distance(a, b, distribution_family="categorical")


def test_fisher_rao_returns_method_key():
    a = {"mean": 0.0, "std": 1.0}
    b = {"mean": 1.0, "std": 2.0}
    r = fisher_rao_distance(a, b, distribution_family="univariate_gaussian")
    assert r["method"] == "closed_form"


def test_fisher_rao_multivariate_method_numerical():
    a = {"mean": np.zeros(2), "cov": np.eye(2)}
    b = {"mean": np.ones(2), "cov": 2.0 * np.eye(2)}
    r = fisher_rao_distance(a, b, distribution_family="multivariate_gaussian")
    assert r["method"] == "numerical"


def test_fisher_rao_unsupported_family_raises():
    a = {"mean": 0.0, "std": 1.0}
    with pytest.raises(NotImplementedError):
        fisher_rao_distance(a, a, distribution_family="weibull")


def test_fisher_rao_multivariate_dim_mismatch_raises():
    a = {"mean": np.zeros(2), "cov": np.eye(2)}
    b = {"mean": np.zeros(3), "cov": np.eye(3)}
    with pytest.raises(ValueError):
        fisher_rao_distance(a, b, distribution_family="multivariate_gaussian")
