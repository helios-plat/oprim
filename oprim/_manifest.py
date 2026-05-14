"""Oprim element manifest — authoritative list of all public elements."""  # pragma: no cover

from __future__ import annotations

VERSION = "1.11.0"

ELEMENTS: list[str] = [
    # Time Series (11)
    "log_returns",
    "cumulative_returns",
    "rolling_window_split",
    "lag_forward_fill",
    "percentile_rank",
    "ewma_smooth",
    "realized_vol",
    "zscore_normalize",
    "gap_detect",
    "resample_align",
    "purge_embargo_split",
    # Statistics (14)
    "bootstrap_ci",
    "percentile_ci",
    "percentile_value",
    "distribution_summary",
    "skew_kurt_robust",
    "kolmogorov_smirnov_test",
    "mann_kendall_trend",
    "bayes_beta_update",
    "brier_score_decomposed",
    "pearson_spearman_corr",
    "kde_density",
    "correlation_batch",
    "bayesian_changepoint",
    "posterior_run_length",
    # Distance (5)
    "wasserstein_distance",
    "dtw_distance",
    "cosine_similarity_batch",
    "euclidean_distance_matrix",
    "symmetric_kl_divergence",
    # Phase 7A:
    "distributional_distance",
    # Numerics (3)
    "logsumexp_safe",
    "softmax_safe",
    "clip_with_warning",
    # Regime (3)
    "regime_filter_data",
    "regime_transition_matrix",
    "regime_label_align",
    # Finance (6)
    "drawdown_curve",
    "sharpe_ratio",
    "beta_alpha_ols",
    "value_at_risk",
    "nelson_siegel_yield_curve",
    "futures_curve_shape",
    # Information (3)
    "shannon_entropy",
    "ordinal_pattern",
    "phase_randomize",
    # Signal Processing (6)
    "linear_slope",
    "atr",
    "hurst_exponent",
    "compute_dwt",
    "H_change_rate_std",
    "orderbook_entropy",
    # Topology (2)
    "takens_embed",
    "persistence_landscape",
    # Point Process (1)
    "hawkes_nll",
    # --- Phase 1 additions (v1.4.0) ---
    # Technical (8)
    "sma",
    "ema",
    "vwap",
    "macd",
    "rsi_normalized",
    "bollinger_bands",
    "donchian_channel",
    "chandelier_exit",
    # Crypto (4)
    "sha256_hash",
    "hmac_sha256",
    "rfc6962_merkle_root",
    "rfc6962_inclusion_proof",
    # Serialization (1)
    "canonical_json",
    # Risk (1)
    "cvar",
    # --- Phase 2 additions (v1.5.0) ---
    # Performance (2)
    "cumulative_returns_perf",
    "cagr",
    # Mean Reversion (2)
    "ornstein_uhlenbeck_fit",
    "ornstein_uhlenbeck_half_life",
    # Volatility (3)
    "garch_fit",
    "garch_forecast",
    "ewma_volatility",
    # Derivatives (3)
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    # --- Phase 3 additions (v1.6.0) ---
    # Similarity (1)
    "vector_similarity",
    # --- Phase 5A additions (v1.8.0) ---
    # Derivatives (8)
    "binomial_tree_price",
    "mc_european_price",
    "mc_asian_price",
    "barrier_option_price",
    "lookback_option_price",
    "lsm_american_price",
    "svensson_yield_curve",
    "cubic_spline_yield_curve",
    # --- Phase 4 additions (v1.7.0) ---
    # Risk (1)
    "mean_deviation",
    # Timeseries (9)
    "adf_test",
    "kpss_test",
    "engle_granger_cointegration",
    "johansen_cointegration",
    "ljung_box_test",
    "durbin_watson",
    "granger_causality_test",
    "jarque_bera_test",
    "breusch_pagan_test",
    # Volatility (8)
    "egarch_fit",
    "egarch_forecast",
    "gjr_garch_fit",
    "gjr_garch_forecast",
    "realized_variance",
    "parkinson_volatility",
    "garman_klass_volatility",
    "yang_zhang_volatility",
    # Technical (7)
    "kama",
    "stochastic_oscillator",
    "cci",
    "williams_r",
    "keltner_channels",
    "obv",
    "mfi",
    # Phase 9A additions (v1.11.0)
    "path_signature_compute",
    "fisher_rao_distance",
    "rough_volatility_simulate",
    "sabr_implied_volatility",
]

CATEGORIES: dict[str, list[str]] = {
    "time_series": [
        "log_returns", "cumulative_returns", "rolling_window_split",
        "lag_forward_fill", "percentile_rank", "ewma_smooth",
        "realized_vol", "zscore_normalize", "gap_detect",
        "resample_align", "purge_embargo_split",
    ],
    "statistics": [
        "bootstrap_ci", "percentile_ci", "percentile_value",
        "distribution_summary", "skew_kurt_robust", "kolmogorov_smirnov_test",
        "mann_kendall_trend", "bayes_beta_update", "brier_score_decomposed",
        "pearson_spearman_corr", "kde_density", "correlation_batch",
        "bayesian_changepoint", "posterior_run_length",
    ],
    "distance": [
        "wasserstein_distance", "dtw_distance", "cosine_similarity_batch",
        "euclidean_distance_matrix", "symmetric_kl_divergence",
        "distributional_distance",
    ],
    "numerics": ["logsumexp_safe", "softmax_safe", "clip_with_warning"],
    "regime": ["regime_filter_data", "regime_transition_matrix", "regime_label_align"],
    "finance": [
        "drawdown_curve", "sharpe_ratio", "beta_alpha_ols", "value_at_risk",
        "nelson_siegel_yield_curve", "futures_curve_shape",
    ],
    "information": ["shannon_entropy", "ordinal_pattern", "phase_randomize"],
    "signal_processing": [
        "linear_slope", "atr", "hurst_exponent", "compute_dwt",
        "H_change_rate_std", "orderbook_entropy",
    ],
    "topology": ["takens_embed", "persistence_landscape"],
    "point_process": ["hawkes_nll"],
    "technical": [
        "sma", "ema", "vwap", "macd", "rsi_normalized",
        "bollinger_bands", "donchian_channel", "chandelier_exit",
        "stochastic_oscillator", "cci", "williams_r",
        "keltner_channels",
        "kama",
        "obv", "mfi",
    ],
    "crypto": [
        "sha256_hash", "hmac_sha256",
        "rfc6962_merkle_root", "rfc6962_inclusion_proof",
        "ed25519_keypair_generate", "ed25519_sign", "ed25519_verify",
    ],
    "serialization": ["canonical_json"],
    "risk": ["cvar", "mean_deviation"],
    # Phase 2 categories
    "performance": ["cumulative_returns_perf", "cagr"],
    "mean_reversion": ["ornstein_uhlenbeck_fit", "ornstein_uhlenbeck_half_life"],
    "volatility": [
        "garch_fit", "garch_forecast", "ewma_volatility",
        "egarch_fit", "egarch_forecast",
        "gjr_garch_fit", "gjr_garch_forecast",
        "realized_variance",
        "parkinson_volatility", "garman_klass_volatility", "yang_zhang_volatility",
        "rough_volatility_simulate",
    ],
    "derivatives": [
        "black_scholes_price", "black_scholes_greeks", "implied_volatility",
        "binomial_tree_price",
        "mc_european_price", "mc_asian_price",
        "barrier_option_price", "lookback_option_price",
        "lsm_american_price",
        "svensson_yield_curve", "cubic_spline_yield_curve",
        "sabr_implied_volatility",
    ],
    # Phase 3 categories
    "similarity": ["vector_similarity"],
    # Phase 9A categories
    "signature": ["path_signature_compute"],
    "info_geometry": ["fisher_rao_distance"],
    # Phase 4 categories
    "timeseries": [
        "adf_test", "kpss_test",
        "engle_granger_cointegration", "johansen_cointegration",
        "ljung_box_test", "durbin_watson",
        "granger_causality_test",
        "jarque_bera_test",
        "breusch_pagan_test",
    ],
}

STABILITY: dict[str, str] = {e: "stable" for e in ELEMENTS}

