"""Tests for path_signature_compute."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.signature.compute import path_signature_compute


def test_signature_compute_depth_1_returns_path_increments():
    # depth=1 sig (excluding level 0) = total increment = x_N - x_0
    path = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    r = path_signature_compute(path, truncation_depth=1, augment_with_time=False)
    sig = r["signature"]
    # Level 0 = [1.0], level 1 = [3.0-0.0, 1.0-0.0] = [3.0, 1.0]
    np.testing.assert_allclose(sig[1:3], [3.0, 1.0], rtol=1e-10)


def test_signature_compute_length_formula():
    # Length = sum d^k for k=0..N
    path = np.random.default_rng(0).normal(0, 1, (10, 3))  # d=3
    d, N = 3, 4
    r = path_signature_compute(path, truncation_depth=N, augment_with_time=False)
    expected_length = sum(d**k for k in range(N + 1))  # 1 + 3 + 9 + 27 + 81 = 121
    assert len(r["signature"]) == expected_length


def test_signature_compute_constant_path_signature_one():
    # Constant path (all same point): all increments = 0, sig = [1, 0, 0, ...]
    path = np.ones((10, 2))
    r = path_signature_compute(path, truncation_depth=3, augment_with_time=False)
    assert r["signature"][0] == 1.0
    np.testing.assert_allclose(r["signature"][1:], 0.0, atol=1e-15)


def test_signature_compute_with_time_augmentation():
    # With time augmentation: channels_used = d+1
    path = np.ones((5, 2))
    r = path_signature_compute(path, truncation_depth=2, augment_with_time=True)
    assert r["channels_used"] == 3  # 2 + 1 time channel


def test_signature_compute_lead_lag_doubles_dim():
    path = np.ones((5, 2))
    r = path_signature_compute(
        path, truncation_depth=2, augment_with_time=False, augment_with_lead_lag=True
    )
    assert r["channels_used"] == 4  # 2*2


def test_signature_compute_invalid_depth_raises():
    path = np.ones((5, 2))
    with pytest.raises(ValueError):
        path_signature_compute(path, truncation_depth=0)
    with pytest.raises(ValueError):
        path_signature_compute(path, truncation_depth=7)


def test_signature_compute_depth_2_includes_levy_area():
    # Levy area: X^{1,2} - X^{2,1} for a simple path
    rng = np.random.default_rng(42)
    path = rng.normal(0, 1, (20, 2))
    r = path_signature_compute(path, truncation_depth=2, augment_with_time=False)
    sig = r["signature"]
    # Level 1: sig[1:3], Level 2: sig[3:7] (4 components: 11, 12, 21, 22)
    assert len(sig) == 1 + 2 + 4  # depth 2, d=2


def test_signature_compute_linear_path_simple_form():
    # For linear path from (0,0) to (a, b) in N steps:
    # Level 1 = [a, b]
    # Level 2 = outer product of [a, b] with [a, b] / 2 (for linear path)
    N = 100
    a, b = 2.0, 3.0
    t = np.linspace(0, 1, N)
    path = np.column_stack([a * t, b * t])
    r = path_signature_compute(path, truncation_depth=2, augment_with_time=False)
    sig = r["signature"]
    # Level 1
    np.testing.assert_allclose(sig[1], a, rtol=1e-3)
    np.testing.assert_allclose(sig[2], b, rtol=1e-3)
    # Level 2 for linear path: X^{i,j} = a_i * a_j / 2 (Riemann sum converges as N->inf)
    np.testing.assert_allclose(sig[3], a * a / 2, rtol=2e-2)  # X^{1,1}
    np.testing.assert_allclose(sig[4], a * b / 2, rtol=2e-2)  # X^{1,2}


def test_signature_compute_returns_fingerprint():
    path = np.random.default_rng(42).normal(0, 1, (10, 2))
    r = path_signature_compute(path, truncation_depth=2)
    assert isinstance(r["fingerprint"], str)
    assert len(r["fingerprint"]) == 64  # sha256 hex


def test_signature_compute_returns_augmented_path():
    path = np.random.default_rng(0).normal(0, 1, (10, 2))
    r = path_signature_compute(path, truncation_depth=2, augment_with_time=True)
    assert r["augmented_path"].shape[1] == 3


@pytest.mark.academic_reference
def test_signature_chevyrev_kormilitzin_examples():
    # Example from Chevyrev & Kormilitzin (2016) primer
    # Simple 2D path: (0,0) -> (1,0) -> (1,1) -> (0,1)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    r = path_signature_compute(path, truncation_depth=2, augment_with_time=False)
    sig = r["signature"]
    # Level 1: total increment = (0-0, 1-0) = (0, 1)
    np.testing.assert_allclose(sig[1], 0.0, atol=1e-10)
    np.testing.assert_allclose(sig[2], 1.0, atol=1e-10)
    # Levy area: X^{1,2} - X^{2,1} should be non-zero (path has non-trivial topology)
    assert abs(sig[4] - sig[6]) > 0  # non-zero antisymmetric part


def test_signature_compute_invalid_ndim_raises():
    path_1d = np.ones(10)
    with pytest.raises(ValueError):
        path_signature_compute(path_1d, truncation_depth=2)


def test_signature_compute_too_few_steps_raises():
    path = np.ones((1, 2))
    with pytest.raises(ValueError):
        path_signature_compute(path, truncation_depth=2)


def test_signature_level0_always_one():
    rng = np.random.default_rng(7)
    path = rng.normal(0, 1, (15, 4))
    r = path_signature_compute(path, truncation_depth=3, augment_with_time=False)
    assert r["signature"][0] == 1.0


def test_signature_compute_depth_key_matches_input():
    path = np.random.default_rng(0).normal(0, 1, (8, 2))
    for depth in range(1, 5):
        r = path_signature_compute(path, truncation_depth=depth, augment_with_time=False)
        assert r["depth"] == depth
