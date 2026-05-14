# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.4.0] - 2026-05-14

### Added (Phase 1: 14 new elements)

#### Technical Indicators (`oprim/technical/`)
- `sma`: Simple Moving Average — SMA_t = (1/N) * sum(P_{t-N+1}..P_t)
- `ema`: Exponential Moving Average (adjust=False/True) — matches pandas ewm exactly
- `vwap`: Volume Weighted Average Price (cumulative and rolling)
- `macd`: MACD line, signal line, histogram — Appel (1979)
- `rsi_normalized`: RSI normalized to [0,1] — Wilder (1978) SMA-seeded smoothing
- `bollinger_bands`: Upper/middle/lower/bandwidth/%B — population std (ddof=0)
- `donchian_channel`: Upper/middle/lower rolling extrema
- `chandelier_exit`: Long/short trailing stop — Le Beau ATR-based

#### Cryptographic Primitives (`oprim/crypto/`)
- `sha256_hash`: NIST FIPS 180-4 SHA-256, returns 64-char hex string, accepts bytes/str
- `hmac_sha256`: RFC 2104 HMAC-SHA-256, returns 64-char hex string
- `rfc6962_merkle_root`: RFC 6962 MTH with arbitrary-bytes leaves (not pre-hashed)
- `rfc6962_inclusion_proof`: RFC 6962 §2.1.1 audit path

#### Serialization (`oprim/serialization/`)
- `canonical_json`: RFC 8785 JCS — deterministic, UTF-16 sorted keys, no whitespace

#### Risk (`oprim/risk/`)
- `cvar`: Conditional Value at Risk (historical & Gaussian closed-form methods)

### Infrastructure
- New subdirectory structure: `technical/`, `crypto/`, `serialization/`, `risk/`
- `oprim/_manifest.py`: authoritative element list with categories and stability tags
- JSON schemas in `oprim/schemas/<category>/<element>.schema.json`
- Private helpers in `oprim/technical/_base.py` (H2 exempt from H1)
- `pyproject.toml`: restored `academic_reference` pytest marker

### Notes
- 433 total tests (144 new Phase 1 tests), 80% coverage
- All Phase 1 elements have `@pytest.mark.academic_reference` tests
- Crypto elements verified against NIST FIPS 180-4, RFC 4231, RFC 6962, RFC 8785
- Technical indicators verified against pandas rolling/ewm reference implementations
- No new third-party dependencies (crypto uses Python stdlib only)

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
