"""Oprim element manifest — authoritative list of all public elements."""

from __future__ import annotations

VERSION = "1.4.0"

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
    ],
    "crypto": [
        "sha256_hash", "hmac_sha256",
        "rfc6962_merkle_root", "rfc6962_inclusion_proof",
    ],
    "serialization": ["canonical_json"],
    "risk": ["cvar"],
}

STABILITY: dict[str, str] = {e: "stable" for e in ELEMENTS}
