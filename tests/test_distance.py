"""Tests for oprim.distance module."""

import numpy as np
import pytest
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from oprim.distance import (
    cosine_similarity_batch,
    dtw_distance,
    euclidean_distance_matrix,
    symmetric_kl_divergence,
    wasserstein_distance,
)


# ============================================================
# wasserstein_distance
# ============================================================
class TestWassersteinDistance:
    def test_same_distribution(self):
        data = np.arange(100.0)
        assert wasserstein_distance(data, data) == pytest.approx(0.0, abs=1e-10)

    def test_shifted(self):
        u = np.arange(100.0)
        v = u + 5.0
        result = wasserstein_distance(u, v)
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_gaussian_known(self):
        """W1 between N(0,1) and N(μ,1) ≈ |μ| for large samples."""
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, 10000)
        v = rng.normal(3, 1, 10000)
        result = wasserstein_distance(u, v)
        assert abs(result - 3.0) < 0.1

    def test_sliced_multi_d(self):
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, (200, 5))
        v = rng.normal(0, 1, (200, 5))
        result = wasserstein_distance(u, v, mode="sliced_multi_d", random_state=42)
        assert result >= 0

    def test_sliced_shifted(self):
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, (500, 3))
        v = rng.normal(5, 1, (500, 3))
        result = wasserstein_distance(u, v, mode="sliced_multi_d", random_state=42)
        assert result > 1.0

    def test_academic_1d_vs_scipy(self):
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, 500)
        v = rng.normal(1, 2, 500)
        result = wasserstein_distance(u, v, mode="1d")
        expected = sp_stats.wasserstein_distance(u, v)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# dtw_distance
# ============================================================
class TestDtwDistance:
    def test_same_sequence(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = dtw_distance(x, x)
        assert result["distance"] == pytest.approx(0.0, abs=1e-10)

    def test_shifted(self):
        x = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0])
        result = dtw_distance(x, y)
        assert result["distance"] >= 0
        assert result["path"] is not None

    def test_with_window(self):
        x = np.arange(20.0)
        y = np.arange(20.0) + 0.5
        result = dtw_distance(x, y, window=5)
        assert result["distance"] >= 0

    def test_multivariate_independent(self):
        x = np.random.default_rng(42).normal(0, 1, (10, 3))
        y = np.random.default_rng(43).normal(0, 1, (10, 3))
        result = dtw_distance(x, y, multivariate_mode="independent")
        assert result["distance"] >= 0

    def test_multivariate_dependent(self):
        x = np.random.default_rng(42).normal(0, 1, (10, 3))
        y = np.random.default_rng(43).normal(0, 1, (10, 3))
        result = dtw_distance(x, y, multivariate_mode="dependent")
        assert result["distance"] >= 0

    def test_manhattan(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        result = dtw_distance(x, y, distance_metric="manhattan")
        assert result["distance"] == pytest.approx(0.0, abs=1e-10)


# ============================================================
# cosine_similarity_batch
# ============================================================
class TestCosineSimilarityBatch:
    def test_identical(self):
        query = np.array([1.0, 0.0, 0.0])
        db = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = cosine_similarity_batch(query, db)
        assert result[0] == pytest.approx(1.0, abs=1e-9)
        assert result[1] == pytest.approx(0.0, abs=1e-9)

    def test_batch_query(self):
        query = np.array([[1.0, 0.0], [0.0, 1.0]])
        db = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = cosine_similarity_batch(query, db)
        assert result.shape == (2, 3)

    def test_top_k(self):
        query = np.array([1.0, 0.0, 0.0])
        db = np.random.default_rng(42).normal(0, 1, (100, 3))
        scores, indices = cosine_similarity_batch(query, db, top_k=5)
        assert len(scores) == 5
        assert len(indices) == 5
        # Scores should be sorted descending
        assert all(scores[i] >= scores[i + 1] for i in range(4))

    def test_pre_normalize(self):
        rng = np.random.default_rng(42)
        query = rng.normal(0, 1, (1, 5))
        db = rng.normal(0, 1, (10, 5))
        # Normalize
        query_n = query / np.linalg.norm(query, axis=1, keepdims=True)
        db_n = db / np.linalg.norm(db, axis=1, keepdims=True)
        r1 = cosine_similarity_batch(query.squeeze(), db)
        r2 = cosine_similarity_batch(query_n.squeeze(), db_n, pre_normalize=True)
        np.testing.assert_allclose(r1, r2, rtol=1e-9)

    def test_academic_vs_sklearn(self):
        rng = np.random.default_rng(42)
        query = rng.normal(0, 1, (3, 10))
        db = rng.normal(0, 1, (20, 10))
        result = cosine_similarity_batch(query, db)
        expected = sklearn_cosine(query, db)
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# euclidean_distance_matrix
# ============================================================
class TestEuclideanDistanceMatrix:
    def test_self_distance(self):
        X = np.array([[0, 0], [1, 0], [0, 1.0]])
        result = euclidean_distance_matrix(X)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[0, 1] == pytest.approx(1.0)

    def test_x_vs_y(self):
        X = np.array([[0, 0.0]])
        Y = np.array([[3, 4.0]])
        result = euclidean_distance_matrix(X, Y)
        assert result[0, 0] == pytest.approx(5.0)

    def test_weighted(self):
        X = np.array([[1, 0.0]])
        Y = np.array([[0, 0.0]])
        w = np.array([4.0, 1.0])
        result = euclidean_distance_matrix(X, Y, weights=w)
        # sqrt(4 * 1^2) = 2
        assert result[0, 0] == pytest.approx(2.0)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0])
        result = euclidean_distance_matrix(X)
        assert result.shape == (3, 3)

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (50, 5))
        Y = rng.normal(0, 1, (30, 5))
        result = euclidean_distance_matrix(X, Y)
        expected = cdist(X, Y, metric="euclidean")
        np.testing.assert_allclose(result, expected, rtol=1e-9)


# ============================================================
# symmetric_kl_divergence
# ============================================================
class TestSymmetricKLDivergence:
    def test_same_distribution(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        result = symmetric_kl_divergence(p, p)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_js_vs_symmetric_kl(self):
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.3])
        js = symmetric_kl_divergence(p, q, mode="js")
        skl = symmetric_kl_divergence(p, q, mode="symmetric_kl")
        # JS is bounded by ln(2), symmetric KL is not
        assert js < skl

    def test_zero_probability(self):
        """Epsilon smoothing handles zeros."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        result = symmetric_kl_divergence(p, q)
        assert np.isfinite(result)

    def test_base_2(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.9, 0.1])
        r_e = symmetric_kl_divergence(p, q, base="e")
        r_2 = symmetric_kl_divergence(p, q, base="2")
        # log2 = ln / ln(2), so result_2 = result_e / ln(2)
        np.testing.assert_allclose(r_2, r_e / np.log(2), rtol=1e-6)

    def test_academic_js_vs_scipy(self):
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.1, 0.5, 0.4])
        result = symmetric_kl_divergence(p, q, mode="js", base="2")
        # scipy jensenshannon returns sqrt(JS), so square it
        expected = jensenshannon(p, q, base=2) ** 2
        np.testing.assert_allclose(result, expected, rtol=1e-4)
