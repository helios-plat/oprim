"""Tests for distributional_distance in oprim.distance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

from oprim.distance import distributional_distance

METRICS = ["wasserstein_1", "kolmogorov_smirnov", "cramer_von_mises", "energy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 1. test_identical_samples_all_metrics_zero
# ---------------------------------------------------------------------------
class TestIdenticalSamplesAllMetricsZero:
    @pytest.mark.parametrize("metric", METRICS)
    def test_identical(self, metric):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = distributional_distance(a, a, metric=metric)
        assert result == pytest.approx(0.0, abs=1e-10), (
            f"metric={metric}: expected 0, got {result}"
        )

    @pytest.mark.parametrize("metric", METRICS)
    def test_identical_large(self, metric):
        a = _rng(1).normal(0, 1, 200)
        result = distributional_distance(a, a, metric=metric)
        assert result == pytest.approx(0.0, abs=1e-9), (
            f"metric={metric}: expected 0, got {result}"
        )


# ---------------------------------------------------------------------------
# 2. test_wasserstein_shifted_distribution  (shift by 1 → W1 = 1.0)
# ---------------------------------------------------------------------------
class TestWassersteinShiftedDistribution:
    def test_shift_by_one(self):
        a = np.arange(100.0)
        b = a + 1.0
        result = distributional_distance(a, b, metric="wasserstein_1")
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_unit_normals_shifted_by_1(self):
        """Two large unit-normal samples shifted by 1 → W1 ≈ 1.0."""
        rng = _rng(0)
        a = rng.normal(0, 1, 50_000)
        b = rng.normal(1, 1, 50_000)
        result = distributional_distance(a, b, metric="wasserstein_1")
        assert abs(result - 1.0) < 0.05


# ---------------------------------------------------------------------------
# 3. test_ks_max_cdf_diff  (manually verify KS statistic)
# ---------------------------------------------------------------------------
class TestKSMaxCDFDiff:
    def test_two_point_distributions(self):
        # a = [0, 0, 1, 1], b = [0.5, 0.5, 0.5, 0.5]
        a = np.array([0.0, 0.0, 1.0, 1.0])
        b = np.array([0.5, 0.5, 0.5, 0.5])
        result = distributional_distance(a, b, metric="kolmogorov_smirnov")
        # F_a at 0 = 0.5, F_b at 0 = 0; diff = 0.5
        # F_a at 0.5 = 0.5, F_b at 0.5 = 1.0; diff = 0.5
        # F_a at 1 = 1.0, F_b at 1 = 1.0; diff = 0.0
        assert result == pytest.approx(0.5, abs=1e-10)

    def test_non_overlapping(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        result = distributional_distance(a, b, metric="kolmogorov_smirnov")
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_ks_nonnegative(self):
        rng = _rng(5)
        a = rng.normal(0, 1, 100)
        b = rng.normal(1, 2, 100)
        result = distributional_distance(a, b, metric="kolmogorov_smirnov")
        assert result >= 0.0
        assert result <= 1.0  # KS stat is bounded in [0, 1]


# ---------------------------------------------------------------------------
# 4. test_cvm_basic
# ---------------------------------------------------------------------------
class TestCvMBasic:
    def test_identical_zero(self):
        a = np.linspace(0, 1, 50)
        result = distributional_distance(a, a, metric="cramer_von_mises")
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_positive(self):
        a = np.array([0.0, 1.0, 2.0, 3.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        result = distributional_distance(a, b, metric="cramer_von_mises")
        assert result > 0.0

    def test_cvm_nonnegative(self):
        rng = _rng(7)
        a = rng.normal(0, 1, 80)
        b = rng.exponential(1, 80)
        result = distributional_distance(a, b, metric="cramer_von_mises")
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 5. test_energy_basic
# ---------------------------------------------------------------------------
class TestEnergyBasic:
    def test_different_distributions_positive(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 10.0, 10.0])
        result = distributional_distance(a, b, metric="energy")
        assert result > 0.0

    def test_energy_known_value(self):
        """E(P, Q) for P=delta(0), Q=delta(1): 2*1 - 0 - 0 = 2.0."""
        a = np.array([0.0])
        b = np.array([1.0])
        result = distributional_distance(a, b, metric="energy")
        assert result == pytest.approx(2.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 6. test_symmetric_all_metrics  (swap a and b → same result)
# ---------------------------------------------------------------------------
class TestSymmetricAllMetrics:
    @pytest.mark.parametrize("metric", METRICS)
    def test_symmetric(self, metric):
        rng = _rng(10)
        a = rng.normal(0, 1, 50)
        b = rng.normal(1, 1.5, 50)
        d_ab = distributional_distance(a, b, metric=metric)
        d_ba = distributional_distance(b, a, metric=metric)
        assert d_ab == pytest.approx(d_ba, rel=1e-9), (
            f"metric={metric}: d(a,b)={d_ab} != d(b,a)={d_ba}"
        )


# ---------------------------------------------------------------------------
# 7. test_weights_wasserstein  (weighted wasserstein with all weight on one element)
# ---------------------------------------------------------------------------
class TestWeightsWasserstein:
    def test_all_weight_on_one_element(self):
        """Effectively reduces each sample to a single point."""
        a = np.array([0.0, 1.0, 2.0])
        b = np.array([5.0, 6.0, 7.0])
        # All weight on a[0]=0.0 and b[0]=5.0
        wa = np.array([1.0, 0.0, 0.0])
        wb = np.array([1.0, 0.0, 0.0])
        result = distributional_distance(a, b, metric="wasserstein_1", weights_a=wa, weights_b=wb)
        # W1 between delta(0) and delta(5) = 5.0
        assert result == pytest.approx(5.0, abs=1e-9)

    def test_weights_ks_equal_weight(self):
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 1.0])
        wa = np.array([1.0, 1.0])
        wb = np.array([1.0, 1.0])
        result = distributional_distance(a, b, metric="kolmogorov_smirnov", weights_a=wa, weights_b=wb)
        assert result == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 8. test_empty_raises
# ---------------------------------------------------------------------------
class TestEmptyRaises:
    def test_empty_sample_a_raises(self):
        with pytest.raises(ValueError, match="sample_a"):
            distributional_distance(np.array([]), np.array([1.0]))

    def test_empty_sample_b_raises(self):
        with pytest.raises(ValueError, match="sample_b"):
            distributional_distance(np.array([1.0]), np.array([]))

    def test_weight_length_mismatch_a(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="weights_a"):
            distributional_distance(a, b, weights_a=np.array([1.0, 1.0]))

    def test_weight_length_mismatch_b(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="weights_b"):
            distributional_distance(a, b, weights_b=np.array([1.0]))


# ---------------------------------------------------------------------------
# 9. test_invalid_metric_raises
# ---------------------------------------------------------------------------
class TestInvalidMetricRaises:
    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            distributional_distance(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                metric="not_a_metric",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# 10. test_returns_float
# ---------------------------------------------------------------------------
class TestReturnsFloat:
    @pytest.mark.parametrize("metric", METRICS)
    def test_return_type_is_float(self, metric):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        result = distributional_distance(a, b, metric=metric)
        assert isinstance(result, float), f"metric={metric}: expected float, got {type(result)}"


# ---------------------------------------------------------------------------
# 11. test_wasserstein_matches_scipy_oracle
# ---------------------------------------------------------------------------
class TestWassersteinMatchesScipy:
    def test_matches_scipy_oracle(self):
        rng = _rng(99)
        a = rng.normal(0, 1, 200)
        b = rng.gamma(2, 2, 200)
        result = distributional_distance(a, b, metric="wasserstein_1")
        expected = sp_stats.wasserstein_distance(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_matches_scipy_with_weights(self):
        rng = _rng(77)
        a = rng.normal(0, 1, 50)
        b = rng.normal(2, 1, 50)
        wa = rng.uniform(0.1, 1.0, 50)
        wb = rng.uniform(0.1, 1.0, 50)
        result = distributional_distance(a, b, metric="wasserstein_1", weights_a=wa, weights_b=wb)
        expected = sp_stats.wasserstein_distance(a, b, wa, wb)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 12. test_energy_szekely_property
# ---------------------------------------------------------------------------
class TestEnergySzekelyProperty:
    def test_self_distance_zero(self):
        """E(X, X) = 0."""
        a = _rng(20).normal(0, 1, 100)
        result = distributional_distance(a, a, metric="energy")
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_positive(self):
        """E(X, Y) > 0 for distributions with different means."""
        a = _rng(21).normal(0, 1, 100)
        b = _rng(22).normal(5, 1, 100)
        result = distributional_distance(a, b, metric="energy")
        assert result > 0.0

    def test_energy_single_points(self):
        """E(delta_x, delta_y) = 2 * |x - y|."""
        x, y = 3.0, 7.0
        a = np.array([x])
        b = np.array([y])
        result = distributional_distance(a, b, metric="energy")
        assert result == pytest.approx(2.0 * abs(x - y), abs=1e-10)


# ---------------------------------------------------------------------------
# 13. pd.Series input
# ---------------------------------------------------------------------------
class TestPdSeriesInput:
    @pytest.mark.parametrize("metric", METRICS)
    def test_series_input(self, metric):
        a = pd.Series([1.0, 2.0, 3.0, 4.0])
        b = pd.Series([2.0, 3.0, 4.0, 5.0])
        result = distributional_distance(a, b, metric=metric)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_series_matches_array(self):
        rng = _rng(55)
        arr_a = rng.normal(0, 1, 30)
        arr_b = rng.normal(1, 1, 30)
        result_arr = distributional_distance(arr_a, arr_b, metric="wasserstein_1")
        result_ser = distributional_distance(
            pd.Series(arr_a), pd.Series(arr_b), metric="wasserstein_1"
        )
        assert result_arr == pytest.approx(result_ser, rel=1e-12)
