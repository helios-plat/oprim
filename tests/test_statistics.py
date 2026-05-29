"""Tests for oprim.statistics module."""

import warnings

import numpy as np
import pytest
from scipy import stats as sp_stats

from oprim.statistics import (
    bayes_beta_update,
    bootstrap_ci,
    brier_score_decomposed,
    correlation_batch,
    distribution_summary,
    kde_density,
    kolmogorov_smirnov_test,
    mann_kendall_trend,
    pearson_spearman_corr,
    percentile_ci,
    percentile_value,
    skew_kurt_robust,
)


# ============================================================
# bootstrap_ci
# ============================================================
class TestBootstrapCI:
    def test_mean_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 1000)
        result = bootstrap_ci(data, np.mean, random_state=42)
        assert 4.8 < result["ci_lower"] < result["ci_upper"] < 5.2
        assert result["method"] == "percentile"

    def test_median(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = bootstrap_ci(data, np.median, random_state=42)
        assert result["ci_lower"] < 0 < result["ci_upper"]

    def test_sharpe_ratio(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, 252)
        sharpe_fn = lambda x: np.mean(x) / np.std(x, ddof=1) * np.sqrt(252)
        result = bootstrap_ci(returns, sharpe_fn, random_state=42)
        assert "ci_lower" in result

    def test_method_basic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = bootstrap_ci(data, np.mean, method="basic", random_state=42)
        assert result["method"] == "basic"

    def test_method_bca(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        result = bootstrap_ci(data, np.mean, method="bca", random_state=42)
        assert result["method"] == "bca"

    def test_reproducibility(self):
        data = np.arange(100.0)
        r1 = bootstrap_ci(data, np.mean, random_state=123)
        r2 = bootstrap_ci(data, np.mean, random_state=123)
        assert r1["ci_lower"] == r2["ci_lower"]

    def test_nan_handling(self):
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = bootstrap_ci(data, np.mean, random_state=42)
        assert not np.isnan(result["point_estimate"])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]), np.mean)

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([np.nan, np.nan]), np.mean)

    def test_confidence_level_invalid(self):
        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci(np.arange(10.0), np.mean, confidence_level=1.5)

    def test_low_n_bootstrap_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bootstrap_ci(np.arange(50.0), np.mean, n_bootstrap=50, random_state=42)
            assert any("unreliable" in str(x.message) for x in w)

    def test_academic_vs_scipy(self):
        """Compare percentile method with scipy.stats.bootstrap."""
        rng = np.random.default_rng(42)
        data = rng.normal(10, 2, 200)
        result = bootstrap_ci(data, np.mean, n_bootstrap=5000, random_state=42)
        # scipy reference
        res = sp_stats.bootstrap(
            (data,), np.mean, n_resamples=5000, random_state=42, method="percentile"
        )
        # Allow 5% tolerance due to random sampling
        np.testing.assert_allclose(result["ci_lower"], res.confidence_interval.low, rtol=0.05)
        np.testing.assert_allclose(result["ci_upper"], res.confidence_interval.high, rtol=0.05)


# ============================================================
# percentile_ci
# ============================================================
class TestPercentileCI:
    def test_basic(self):
        samples = np.arange(100.0)
        result = percentile_ci(samples)
        assert result["q_0.5"] == pytest.approx(49.5, abs=1)

    def test_custom_quantiles(self):
        samples = np.arange(1000.0)
        result = percentile_ci(samples, quantiles=[0.1, 0.9])
        assert "q_0.1" in result
        assert "q_0.9" in result

    def test_empty_returns_nan(self):
        result = percentile_ci(np.array([]))
        assert np.isnan(result["q_0.5"])

    def test_all_nan_returns_nan(self):
        result = percentile_ci(np.array([np.nan, np.nan]))
        assert np.isnan(result["q_0.5"])

    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError, match="not in"):
            percentile_ci(np.arange(10.0), quantiles=[1.5])

    def test_academic_vs_numpy(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)
        result = percentile_ci(data, quantiles=[0.05, 0.5, 0.95])
        np.testing.assert_allclose(result["q_0.05"], np.percentile(data, 5), rtol=1e-9)
        np.testing.assert_allclose(result["q_0.95"], np.percentile(data, 95), rtol=1e-9)


# ============================================================
# distribution_summary
# ============================================================
class TestDistributionSummary:
    def test_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        result = distribution_summary(data)
        assert abs(result["mean"]) < 0.05
        assert abs(result["std"] - 1.0) < 0.05
        assert result["n"] == 10000

    def test_with_nan(self):
        data = np.array([1.0, 2.0, np.nan, 4.0])
        result = distribution_summary(data)
        assert result["n"] == 3
        assert result["n_nan"] == 1

    def test_empty(self):
        result = distribution_summary(np.array([np.nan, np.nan]))
        assert result["n"] == 0
        assert np.isnan(result["mean"])

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5, 2, 500)
        result = distribution_summary(data)
        np.testing.assert_allclose(result["mean"], np.mean(data), rtol=1e-9)
        expected_skew = sp_stats.skew(data, bias=False)
        np.testing.assert_allclose(result["skew"], expected_skew, rtol=1e-9)


# ============================================================
# skew_kurt_robust
# ============================================================
class TestSkewKurtRobust:
    def test_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        result = skew_kurt_robust(data)
        assert abs(result["skewness"]) < 0.1
        assert abs(result["kurtosis_excess"]) < 0.1

    def test_bias_true_vs_false(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        biased = skew_kurt_robust(data, bias=True)
        unbiased = skew_kurt_robust(data, bias=False)
        assert biased["skewness"] != unbiased["skewness"]

    def test_n_lt_3(self):
        result = skew_kurt_robust(np.array([1.0, 2.0]))
        assert np.isnan(result["skewness"])

    def test_n_lt_4(self):
        result = skew_kurt_robust(np.array([1.0, 2.0, 3.0]))
        assert not np.isnan(result["skewness"])
        assert np.isnan(result["kurtosis_excess"])

    def test_nan_raise(self):
        with pytest.raises(ValueError, match="NaN"):
            skew_kurt_robust(np.array([1.0, np.nan, 3.0]), nan_policy="raise")

    def test_nan_omit(self):
        result = skew_kurt_robust(np.array([1.0, np.nan, 2.0, 3.0, 4.0, 5.0]))
        assert not np.isnan(result["skewness"])

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200)
        result = skew_kurt_robust(data, bias=False)
        expected_skew = sp_stats.skew(data, bias=False)
        expected_kurt = sp_stats.kurtosis(data, fisher=True, bias=False)
        np.testing.assert_allclose(result["skewness"], expected_skew, rtol=1e-9)
        np.testing.assert_allclose(result["kurtosis_excess"], expected_kurt, rtol=1e-9)


# ============================================================
# kolmogorov_smirnov_test
# ============================================================
class TestKolmogorovSmirnovTest:
    def test_two_sample_same(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        result = kolmogorov_smirnov_test(a, b)
        assert result["p_value"] > 0.05

    def test_two_sample_different(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(5, 1, 500)
        result = kolmogorov_smirnov_test(a, b)
        assert result["p_value"] < 0.01

    def test_one_sample_normal(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = kolmogorov_smirnov_test(data, "norm", mode="one_sample")
        assert result["p_value"] > 0.05

    def test_no_sample_b_raises(self):
        with pytest.raises(ValueError, match="sample_b"):
            kolmogorov_smirnov_test(np.arange(10.0), None, mode="two_sample")

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0.5, 1, 200)
        result = kolmogorov_smirnov_test(a, b)
        stat, p = sp_stats.ks_2samp(a, b)
        np.testing.assert_allclose(result["statistic"], stat, rtol=1e-9)
        np.testing.assert_allclose(result["p_value"], p, rtol=1e-9)


# ============================================================
# mann_kendall_trend
# ============================================================
class TestMannKendallTrend:
    def test_increasing(self):
        data = np.arange(50.0)
        result = mann_kendall_trend(data)
        assert result["trend"] == "increasing"
        assert result["p_value"] < 0.05

    def test_decreasing(self):
        data = np.arange(50.0)[::-1]
        result = mann_kendall_trend(data)
        assert result["trend"] == "decreasing"

    def test_no_trend(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 100)
        result = mann_kendall_trend(data)
        assert result["trend"] == "no_trend"

    def test_short_series(self):
        result = mann_kendall_trend(np.array([1.0, 2.0]))
        assert result["trend"] == "no_trend"
        assert result["n"] == 2

    def test_tau_range(self):
        data = np.arange(20.0)
        result = mann_kendall_trend(data)
        assert -1 <= result["tau"] <= 1

    def test_no_correction(self):
        data = np.arange(50.0)
        result = mann_kendall_trend(data, hamed_rao_correction=False)
        assert result["trend"] == "increasing"

    def test_academic_vs_pymannkendall(self):
        """Compare with pyMannKendall if available."""
        try:
            import pymannkendall as mk
            rng = np.random.default_rng(42)
            data = rng.normal(0.001, 0.02, 100)
            # Without correction
            result = mann_kendall_trend(data, hamed_rao_correction=False)
            mk_result = mk.original_test(data)
            np.testing.assert_allclose(result["tau"], mk_result.Tau, rtol=1e-6)
            np.testing.assert_allclose(result["p_value"], mk_result.p, rtol=1e-6)
            # With Hamed-Rao
            result_hr = mann_kendall_trend(data, hamed_rao_correction=True)
            mk_hr = mk.hamed_rao_modification_test(data)
            np.testing.assert_allclose(result_hr["tau"], mk_hr.Tau, rtol=1e-5)
            np.testing.assert_allclose(result_hr["p_value"], mk_hr.p, rtol=1e-4)
        except ImportError:
            pass  # Skip if pyMannKendall not installed


# ============================================================
# bayes_beta_update
# ============================================================
class TestBayesBetaUpdate:
    def test_uniform_prior(self):
        result = bayes_beta_update(1.0, 1.0, 5, 0)
        assert result["posterior_alpha"] == 6.0
        assert result["posterior_beta"] == 1.0
        assert result["posterior_mean"] == pytest.approx(6 / 7)

    def test_informative_prior(self):
        result = bayes_beta_update(10.0, 5.0, 3, 2)
        assert result["posterior_alpha"] == 13.0
        assert result["posterior_beta"] == 7.0

    def test_mode_boundary(self):
        """Mode undefined when alpha < 1 or beta < 1."""
        result = bayes_beta_update(0.5, 0.5, 0, 0)
        assert np.isnan(result["posterior_mode"])

    def test_invalid_prior_raises(self):
        with pytest.raises(ValueError, match="prior"):
            bayes_beta_update(0, 1, 1, 1)

    def test_negative_counts_raises(self):
        with pytest.raises(ValueError, match="successes"):
            bayes_beta_update(1, 1, -1, 0)

    def test_academic_vs_scipy(self):
        result = bayes_beta_update(2.0, 3.0, 10, 5)
        dist = sp_stats.beta(12, 8)
        np.testing.assert_allclose(result["posterior_mean"], dist.mean(), rtol=1e-9)
        np.testing.assert_allclose(result["q_0.5"], dist.ppf(0.5), rtol=1e-9)


# ============================================================
# brier_score_decomposed
# ============================================================
class TestBrierScoreDecomposed:
    def test_perfect_forecast(self):
        forecasts = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1, 0, 1, 0])
        result = brier_score_decomposed(forecasts, outcomes)
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_forecast(self):
        outcomes = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        forecasts = np.full(10, 0.5)
        result = brier_score_decomposed(forecasts, outcomes)
        assert result["brier_score"] == pytest.approx(0.25, abs=1e-10)

    def test_invalid_outcomes_raises(self):
        with pytest.raises(ValueError, match="binary"):
            brier_score_decomposed(np.array([0.5]), np.array([0.5]))

    def test_clip_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            brier_score_decomposed(np.array([1.5, -0.1]), np.array([1, 0]))
            assert any("clipped" in str(x.message) for x in w)

    def test_uncertainty(self):
        """Uncertainty = obar * (1 - obar)."""
        outcomes = np.array([1, 1, 0, 0, 0])  # obar = 0.4
        forecasts = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = brier_score_decomposed(forecasts, outcomes)
        assert result["uncertainty"] == pytest.approx(0.4 * 0.6, abs=1e-10)


# ============================================================
# pearson_spearman_corr
# ============================================================
class TestPearsonSpearmanCorr:
    def test_linear(self):
        x = np.arange(100.0)
        y = 2 * x + 1
        result = pearson_spearman_corr(x, y, min_samples=10)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-9)
        assert result["spearman_r"] == pytest.approx(1.0, abs=1e-9)

    def test_monotone_nonlinear(self):
        x = np.arange(1, 101.0)
        y = np.log(x)
        result = pearson_spearman_corr(x, y, min_samples=10)
        assert result["spearman_r"] == pytest.approx(1.0, abs=1e-9)
        assert result["pearson_r"] < 1.0

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0, 1, 1000)
        result = pearson_spearman_corr(x, y, min_samples=10)
        assert abs(result["pearson_r"]) < 0.1

    def test_min_samples_raises(self):
        with pytest.raises(ValueError, match="samples"):
            pearson_spearman_corr(np.arange(5.0), np.arange(5.0), min_samples=30)

    def test_nan_raise(self):
        with pytest.raises(ValueError, match="NaN"):
            pearson_spearman_corr(
                np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]),
                min_samples=2, nan_policy="raise"
            )

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = x + rng.normal(0, 0.5, 100)
        result = pearson_spearman_corr(x, y, min_samples=10)
        pr, pp = sp_stats.pearsonr(x, y)
        sr, sp = sp_stats.spearmanr(x, y)
        np.testing.assert_allclose(result["pearson_r"], pr, rtol=1e-9)
        np.testing.assert_allclose(result["spearman_r"], sr, rtol=1e-9)


# ============================================================
# kde_density
# ============================================================
class TestKdeDensity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 500)
        result = kde_density(data)
        assert "x" in result
        assert "density" in result
        assert len(result["x"]) == 200

    def test_custom_eval_points(self):
        data = np.arange(100.0)
        pts = np.linspace(0, 100, 50)
        result = kde_density(data, eval_points=pts)
        assert len(result["x"]) == 50

    def test_bandwidth_float(self):
        data = np.arange(100.0)
        result = kde_density(data, bandwidth=0.5)
        assert result["density"].sum() > 0

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="2 data points"):
            kde_density(np.array([1.0]))

    def test_academic_vs_scipy(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 300)
        pts = np.linspace(-3, 3, 50)
        result = kde_density(data, eval_points=pts, bandwidth="scott")
        kde_ref = sp_stats.gaussian_kde(data, bw_method="scott")
        expected = kde_ref(pts)
        np.testing.assert_allclose(result["density"], expected, rtol=1e-9)


# ============================================================
# Additional gap-filling tests (Phase 2)
# ============================================================

class TestBootstrapCIExtraGaps:
    def test_bootstrap_ci_50pct_nan_raises(self):
        """statistic_fn returning NaN for >50% samples raises ValueError."""
        call_count = [0]

        def always_nan(x):
            call_count[0] += 1
            return float("nan")

        with pytest.raises(ValueError, match="50%"):
            bootstrap_ci(np.arange(50.0), always_nan, n_bootstrap=200, random_state=0)

    def test_bootstrap_ci_unknown_method_raises(self):
        """Unknown bootstrap method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            bootstrap_ci(np.arange(50.0), np.mean, method="jackknife", random_state=0)


class TestNanPolicyOmit:
    def test_skew_kurt_nan_policy_omit(self):
        """nan_policy='omit' removes NaN before computation."""
        data_with_nan = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        result = skew_kurt_robust(data_with_nan, nan_policy="omit")
        data_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = skew_kurt_robust(data_clean, nan_policy="omit")
        assert result["skewness"] == pytest.approx(expected["skewness"], rel=1e-9)

    def test_pearson_spearman_nan_policy_omit(self):
        """nan_policy='omit' removes paired NaN before correlation."""
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0] * 10)
        y = np.array([2.0, 4.0, np.nan, 8.0, 10.0] * 10)
        result = pearson_spearman_corr(x, y, min_samples=10, nan_policy="omit")
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-6)


class TestMannKendallExtraGaps:
    def test_mann_kendall_s_negative(self):
        """Strictly decreasing series -> s < 0, trend='decreasing'."""
        data = np.arange(50.0)[::-1]
        result = mann_kendall_trend(data)
        assert result["z_score"] < 0
        assert result["trend"] == "decreasing"

    def test_mann_kendall_s_zero(self):
        """Constant series -> z=0, trend='no_trend'."""
        data = np.full(20, 5.0)
        result = mann_kendall_trend(data)
        assert result["z_score"] == pytest.approx(0.0, abs=1e-9)
        assert result["trend"] == "no_trend"


class TestBrierScoreExtraGaps:
    def test_brier_score_binless(self):
        """method='binless' uses brier_score - uncertainty as reliability."""
        forecasts = np.array([0.8, 0.2, 0.7, 0.3])
        outcomes = np.array([1, 0, 1, 0])
        result = brier_score_decomposed(forecasts, outcomes, method="binless")
        assert "brier_score" in result
        assert result["resolution"] == pytest.approx(0.0, abs=1e-12)
        # reliability = brier_score - uncertainty
        expected_reliability = result["brier_score"] - result["uncertainty"]
        assert result["reliability"] == pytest.approx(expected_reliability, abs=1e-10)


class TestCorrelationBatch:
    def test_correlation_batch_pearson(self):
        """correlation_batch with pearson returns m x m DataFrame."""
        import pandas as pd
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(100, 4)), columns=list("abcd"))
        result = correlation_batch(df, method="pearson")
        assert result.shape == (4, 4)
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(result.values), np.ones(4), rtol=1e-9)

    def test_correlation_batch_spearman(self):
        """correlation_batch with spearman returns m x m DataFrame."""
        import pandas as pd
        rng = np.random.default_rng(42)
        df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
        result = correlation_batch(df, method="spearman")
        assert result.shape == (3, 3)

    def test_correlation_batch_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="method"):
            correlation_batch(df, method="kendall")

    def test_correlation_batch_non_dataframe_raises(self):
        """Non-DataFrame input raises ValueError."""
        with pytest.raises(ValueError, match="DataFrame"):
            correlation_batch(np.eye(3), method="pearson")


class TestPercentileValue:
    def test_rolling_quantile_basic(self):
        """Rolling quantile with window returns array of correct shape."""
        data = np.arange(20, dtype=float)
        result = percentile_value(data, q=0.5, window=10)
        assert len(result) == len(data)
        # Values before window should be NaN
        assert np.isnan(result[:10]).all()

    def test_rolling_quantile_values(self):
        """Check rolling median matches numpy."""
        data = np.arange(20, dtype=float)
        result = percentile_value(data, q=0.5, window=10)
        # At index 19 (last): window is [9..18] (backward-exclusive), median = 13.5
        assert result[19] == pytest.approx(13.5, abs=0.1)

    def test_scalar_quantile(self):
        """No window -> scalar quantile over all data."""
        data = np.arange(100, dtype=float)
        result = percentile_value(data, q=0.5)
        assert result == pytest.approx(np.median(data), rel=1e-9)

    def test_invalid_q_raises(self):
        """q outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="q must be in"):
            percentile_value(np.arange(10.0), q=1.5)
