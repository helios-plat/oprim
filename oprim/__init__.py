"""Oprim — atomic operations library (Layer 1 meta-primitives)."""

from oprim._version import __version__
from oprim.distance import (
    cosine_similarity_batch,
    dtw_distance,
    euclidean_distance_matrix,
    symmetric_kl_divergence,
    wasserstein_distance,
)
from oprim.finance import beta_alpha_ols, drawdown_curve, futures_curve_shape, linear_time_decay, nelson_siegel_yield_curve, sharpe_ratio, trade_metrics, value_at_risk
from oprim.information import ordinal_pattern, phase_randomize, shannon_entropy
from oprim.numerics import clip_with_warning, logsumexp_safe, softmax_safe
from oprim.point_process import hawkes_nll
from oprim.regime import regime_filter_data, regime_label_align, regime_transition_matrix
from oprim.signal_processing import (
    H_change_rate_std,
    atr,
    compute_dwt,
    hurst_exponent,
    linear_slope,
    orderbook_entropy,
)
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
from oprim.time_series import (
    cumulative_returns,
    ewma_smooth,
    gap_detect,
    lag_forward_fill,
    log_returns,
    percentile_rank,
    purge_embargo_split,
    realized_vol,
    resample_align,
    rolling_window_split,
    zscore_normalize,
)
from oprim.topology import persistence_landscape, takens_embed

__all__ = [
    "__version__",
    # Time Series (11)
    "log_returns", "cumulative_returns", "rolling_window_split",
    "lag_forward_fill", "percentile_rank", "ewma_smooth",
    "realized_vol", "zscore_normalize", "gap_detect",
    "resample_align", "purge_embargo_split",
    # Statistics (11)
    "bootstrap_ci", "percentile_ci", "percentile_value",
    "distribution_summary",
    "skew_kurt_robust", "kolmogorov_smirnov_test", "mann_kendall_trend",
    "bayes_beta_update", "brier_score_decomposed", "pearson_spearman_corr",
    "kde_density", "correlation_batch",
    # Distance (5)
    "wasserstein_distance", "dtw_distance", "cosine_similarity_batch",
    "euclidean_distance_matrix", "symmetric_kl_divergence",
    # Numerics (3)
    "logsumexp_safe", "softmax_safe", "clip_with_warning",
    # Regime (3)
    "regime_filter_data", "regime_transition_matrix", "regime_label_align",
    # Finance (4)
    "drawdown_curve", "sharpe_ratio", "beta_alpha_ols", "value_at_risk",
    "nelson_siegel_yield_curve", "futures_curve_shape",
    # Information (3)
    "shannon_entropy", "ordinal_pattern", "phase_randomize",
    # Signal Processing (6)
    "linear_slope", "atr", "hurst_exponent", "compute_dwt",
    "H_change_rate_std", "orderbook_entropy",
    # Topology (2)
    "takens_embed", "persistence_landscape",
    # Point Process (1)
    "hawkes_nll",
]
