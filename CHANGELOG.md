# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-05-10

### Added
- 36 atomic operations across 6 modules:
  - **time_series** (11 ops): log_returns, cumulative_returns, rolling_window_split, lag_forward_fill, percentile_rank, ewma_smooth, realized_vol, zscore_normalize, gap_detect, resample_align, purge_embargo_split
  - **statistics** (10 ops): bootstrap_ci, percentile_ci, distribution_summary, skew_kurt_robust, kolmogorov_smirnov_test, mann_kendall_trend, bayes_beta_update, brier_score_decomposed, pearson_spearman_corr, kde_density
  - **distance** (5 ops): wasserstein_distance, dtw_distance, cosine_similarity_batch, euclidean_distance_matrix, symmetric_kl_divergence
  - **numerics** (3 ops): logsumexp_safe, softmax_safe, clip_with_warning
  - **regime** (3 ops): regime_filter_data, regime_transition_matrix, regime_label_align
  - **finance** (4 ops): drawdown_curve, sharpe_ratio, beta_alpha_ols, value_at_risk

### Fixed
- time_series: zscore_normalize min_periods capping, purge_embargo_split edge cases
- statistics: distribution_summary skew calculation alignment with scipy
- distance: dtw_distance early termination optimization
- finance: sharpe_ratio near-zero std handling, value_at_risk method validation
- regime: stationary distribution for non-ergodic chains, ffill performance optimization, soft mode vectorization
- numerics: logsumexp_safe docstring clarification, clip_with_warning scalar detection

### Testing
- 289 tests with 91% code coverage
- Academic validation tests vs scipy, statsmodels, hmmlearn
- Performance benchmarks for critical paths
