"""Tests for oprim.similarity.vector_similarity.

All mathematical identities are verified to floating-point precision (rtol=1e-7
unless stated otherwise).

Academic reference test requires sklearn (optional; skipped if not installed).
"""
from __future__ import annotations

import numpy as np
import pytest

from oprim.similarity.vector import vector_similarity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of *v*."""
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


# ---------------------------------------------------------------------------
# Test 1: cosine of identical vectors → 1.0
# ---------------------------------------------------------------------------

def test_vector_similarity_cosine_identical_returns_one():
    q = np.array([1.0, 2.0, 3.0])
    C = np.array([[1.0, 2.0, 3.0]])
    result = vector_similarity(q, C, metric="cosine")
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [1.0], rtol=1e-7)


# ---------------------------------------------------------------------------
# Test 2: cosine of orthogonal vectors → 0.0
# ---------------------------------------------------------------------------

def test_vector_similarity_cosine_orthogonal_returns_zero():
    q = np.array([1.0, 0.0])
    C = np.array([[0.0, 1.0]])
    result = vector_similarity(q, C, metric="cosine")
    np.testing.assert_allclose(result, [0.0], atol=1e-15)


# ---------------------------------------------------------------------------
# Test 3: cosine of opposite vectors → -1.0
# ---------------------------------------------------------------------------

def test_vector_similarity_cosine_opposite_returns_minus_one():
    q = np.array([1.0, 2.0, 3.0])
    C = np.array([[-1.0, -2.0, -3.0]])
    result = vector_similarity(q, C, metric="cosine")
    np.testing.assert_allclose(result, [-1.0], rtol=1e-7)


# ---------------------------------------------------------------------------
# Test 4: dot without normalize scales with magnitude
# ---------------------------------------------------------------------------

def test_vector_similarity_dot_unnormalized_scales_with_magnitude():
    q = np.array([1.0, 0.0])
    C = np.array([[2.0, 0.0], [4.0, 0.0]])
    result = vector_similarity(q, C, metric="dot", normalize=False)
    # raw dot products: 2.0 and 4.0
    np.testing.assert_allclose(result, [2.0, 4.0], rtol=1e-7)
    # ratio should be 2× — proves magnitude is not removed
    assert abs(result[1] / result[0] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Test 5: dot with normalize=True equals cosine
# ---------------------------------------------------------------------------

def test_vector_similarity_dot_normalized_equals_cosine():
    rng = np.random.default_rng(42)
    q = rng.standard_normal(8)
    C = rng.standard_normal((20, 8))
    cosine_result = vector_similarity(q, C, metric="cosine", normalize=True)
    dot_result = vector_similarity(q, C, metric="dot", normalize=True)
    np.testing.assert_allclose(dot_result, cosine_result, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 6: euclidean — identical vectors score 0.0 (negated distance = 0)
# ---------------------------------------------------------------------------

def test_vector_similarity_euclidean_basic():
    q = np.array([1.0, 2.0, 3.0])
    C = np.array([[1.0, 2.0, 3.0],   # identical  → −0 = 0
                  [2.0, 2.0, 3.0]])  # differs by 1 in dim 0 → −1
    result = vector_similarity(q, C, metric="euclidean")
    np.testing.assert_allclose(result, [0.0, -1.0], rtol=1e-7)


# ---------------------------------------------------------------------------
# Test 7: manhattan — identical vectors score 0.0 (negated distance = 0)
# ---------------------------------------------------------------------------

def test_vector_similarity_manhattan_basic():
    q = np.array([1.0, 2.0, 3.0])
    C = np.array([[1.0, 2.0, 3.0],   # identical  → −0 = 0
                  [2.0, 3.0, 4.0]])  # differs by 1 in each → −3
    result = vector_similarity(q, C, metric="manhattan")
    np.testing.assert_allclose(result, [0.0, -3.0], rtol=1e-7)


# ---------------------------------------------------------------------------
# Test 8: shape mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_vector_similarity_shape_mismatch_raises():
    q = np.array([1.0, 2.0, 3.0])     # D=3
    C = np.array([[1.0, 2.0]])        # D=2  → mismatch
    with pytest.raises(ValueError, match="[Dd]imension"):
        vector_similarity(q, C)


# ---------------------------------------------------------------------------
# Test 9: invalid metric raises ValueError
# ---------------------------------------------------------------------------

def test_vector_similarity_invalid_metric_raises():
    q = np.array([1.0, 2.0])
    C = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="[Mm]etric"):
        vector_similarity(q, C, metric="chebyshev")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 10: output shape is (N,)
# ---------------------------------------------------------------------------

def test_vector_similarity_output_shape():
    rng = np.random.default_rng(7)
    q = rng.standard_normal(16)
    C = rng.standard_normal((50, 16))
    for metric in ("cosine", "dot", "euclidean", "manhattan"):
        out = vector_similarity(q, C, metric=metric)  # type: ignore[arg-type]
        assert out.shape == (50,), f"metric={metric}: expected (50,), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 11: cosine matches sklearn (academic reference; rtol=1e-12)
# ---------------------------------------------------------------------------

@pytest.mark.academic_reference
def test_vector_similarity_cosine_matches_sklearn():
    sklearn_pairwise = pytest.importorskip("sklearn.metrics.pairwise")
    cosine_similarity = sklearn_pairwise.cosine_similarity

    rng = np.random.default_rng(2024)
    q = rng.standard_normal(32)
    C = rng.standard_normal((100, 32))

    oprim_result = vector_similarity(q, C, metric="cosine", normalize=True)
    # sklearn returns (1, N); flatten to (N,)
    sklearn_result = cosine_similarity(q.reshape(1, -1), C).flatten()

    np.testing.assert_allclose(oprim_result, sklearn_result, rtol=1e-12)


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------

def test_vector_similarity_query_ndim_not_1_raises():
    q = np.array([[1.0, 2.0]])  # 2-D — must raise
    C = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="1-D"):
        vector_similarity(q, C)


def test_vector_similarity_corpus_ndim_not_2_raises():
    q = np.array([1.0, 2.0])
    C = np.array([1.0, 2.0])  # 1-D — must raise
    with pytest.raises(ValueError, match="2-D"):
        vector_similarity(q, C)


def test_vector_similarity_euclidean_ordering():
    """Closest corpus row must have the highest (least negative) score."""
    q = np.array([0.0, 0.0])
    C = np.array([[1.0, 0.0],   # dist = 1
                  [10.0, 0.0],  # dist = 10
                  [0.1, 0.0]])  # dist = 0.1  — closest
    result = vector_similarity(q, C, metric="euclidean")
    assert np.argmax(result) == 2, f"Expected index 2 to be closest, got {np.argmax(result)}"


def test_vector_similarity_manhattan_ordering():
    """Closest corpus row must have the highest (least negative) score."""
    q = np.array([0.0, 0.0])
    C = np.array([[1.0, 1.0],    # L1 = 2
                  [5.0, 5.0],    # L1 = 10
                  [0.1, 0.1]])   # L1 = 0.2  — closest
    result = vector_similarity(q, C, metric="manhattan")
    assert np.argmax(result) == 2, f"Expected index 2 to be closest, got {np.argmax(result)}"


def test_vector_similarity_single_corpus_row():
    """Works correctly when corpus has exactly 1 row."""
    q = np.array([3.0, 4.0])
    C = np.array([[3.0, 4.0]])
    result = vector_similarity(q, C, metric="cosine")
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [1.0], rtol=1e-7)
