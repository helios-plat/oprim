"""Oprim — atomic operations library (Layer 1 meta-primitives)."""

from oprim._version import __version__

# Phase 6A additions (v1.9.0)
from oprim.crypto.ed25519 import ed25519_keypair_generate, ed25519_sign, ed25519_verify
from oprim.crypto.hashing import hmac_sha256, sha256_hash
from oprim.crypto.merkle import rfc6962_inclusion_proof, rfc6962_merkle_root
from oprim.derivatives.american import lsm_american_price

# Phase 5A additions (v1.8.0)
from oprim.derivatives.binomial_tree import binomial_tree_price
from oprim.derivatives.black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from oprim.derivatives.exotic import barrier_option_price, lookback_option_price
from oprim.derivatives.monte_carlo import mc_asian_price, mc_european_price
from oprim.derivatives.rates import cubic_spline_yield_curve, svensson_yield_curve
from oprim.derivatives.sabr import sabr_implied_volatility
from oprim.distance import (
    cosine_similarity_batch,
    distributional_distance,
    dtw_distance,
    euclidean_distance_matrix,
    symmetric_kl_divergence,
    wasserstein_distance,
)
from oprim.finance import (
    beta_alpha_ols,
    drawdown_curve,
    futures_curve_shape,
    nelson_siegel_yield_curve,
    sharpe_ratio,
    value_at_risk,
)
from oprim.info_geometry.fisher_rao import fisher_rao_distance
from oprim.information import ordinal_pattern, phase_randomize, shannon_entropy
from oprim.mean_reversion.ornstein_uhlenbeck import (
    ornstein_uhlenbeck_fit,
    ornstein_uhlenbeck_half_life,
)
from oprim.numerics import clip_with_warning, logsumexp_safe, softmax_safe
from oprim.performance.annualization import cagr

# Phase 2 additions (v1.5.0)
# Note: cumulative_returns below shadows the time_series one;
# the time_series version remains accessible via oprim.time_series directly.
from oprim.performance.cumulative import cumulative_returns
from oprim.point_process import hawkes_nll
from oprim.regime import regime_filter_data, regime_label_align, regime_transition_matrix
from oprim.risk.cvar import cvar

# Phase 4 additions (v1.7.0)
from oprim.risk.dispersion import mean_deviation
from oprim.serialization.canonical import canonical_json
from oprim.signal_processing import (
    H_change_rate_std,
    atr,
    compute_dwt,
    hurst_exponent,
    linear_slope,
    orderbook_entropy,
)

# Phase 9A additions (v1.11.0)
from oprim.signature.compute import path_signature_compute

# Phase 10 additions (v2.0.0)
from oprim.behavioral import (
    cpt_value_function,
    large_loss_aversion_degree,
    probability_weighting_function,
    salience_function,
    salience_ranking_weights,
)
from oprim.recursive_utility import epstein_zin_aggregator
from oprim.spectral import (
    ledoit_wolf_shrinkage,
    marchenko_pastur_threshold,
    rotationally_invariant_estimator,
    spectral_eigengap_detect,
)

# Phase 3 additions (v1.6.0)
from oprim.similarity.vector import vector_similarity
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
from oprim.technical.adaptive import kama
from oprim.technical.bands import bollinger_bands, donchian_channel, keltner_channels
from oprim.technical.exits import chandelier_exit

# Phase 1 additions (v1.4.0)
from oprim.technical.moving_averages import ema, macd, sma, vwap
from oprim.technical.oscillators import cci, rsi_normalized, stochastic_oscillator, williams_r
from oprim.technical.volume import mfi, obv
from oprim.time_series import (
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
from oprim.timeseries.autocorrelation import durbin_watson, ljung_box_test
from oprim.timeseries.causality import granger_causality_test
from oprim.timeseries.cointegration import engle_granger_cointegration, johansen_cointegration
from oprim.timeseries.distribution_tests import jarque_bera_test
from oprim.timeseries.heteroskedasticity import breusch_pagan_test
from oprim.timeseries.stationarity import adf_test, kpss_test
from oprim.topology import persistence_landscape, takens_embed
from oprim.volatility.egarch import egarch_fit, egarch_forecast
from oprim.volatility.ewma import ewma_volatility
from oprim.volatility.garch import garch_fit, garch_forecast
from oprim.volatility.gjr_garch import gjr_garch_fit, gjr_garch_forecast
from oprim.volatility.range_based import (
    garman_klass_volatility,
    parkinson_volatility,
    yang_zhang_volatility,
)
from oprim.volatility.realized import realized_variance
from oprim.volatility.rough import rough_volatility_simulate

__all__ = [
    "__version__",
    # Time Series (11)
    "log_returns", "cumulative_returns", "rolling_window_split",
    "lag_forward_fill", "percentile_rank", "ewma_smooth",
    "realized_vol", "zscore_normalize", "gap_detect",
    "resample_align", "purge_embargo_split",
    # Statistics (12)
    "bootstrap_ci", "percentile_ci", "percentile_value",
    "distribution_summary", "skew_kurt_robust", "kolmogorov_smirnov_test",
    "mann_kendall_trend", "bayes_beta_update", "brier_score_decomposed",
    "pearson_spearman_corr", "kde_density", "correlation_batch",
    # Distance (6)
    "wasserstein_distance", "dtw_distance", "cosine_similarity_batch",
    "euclidean_distance_matrix", "symmetric_kl_divergence",
    "distributional_distance",
    # Numerics (3)
    "logsumexp_safe", "softmax_safe", "clip_with_warning",
    # Regime (3)
    "regime_filter_data", "regime_transition_matrix", "regime_label_align",
    # Finance (6)
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
    # Technical (8) — Phase 1
    "sma", "ema", "vwap", "macd",
    "rsi_normalized",
    "bollinger_bands", "donchian_channel",
    "chandelier_exit",
    # Crypto (4) — Phase 1
    "sha256_hash", "hmac_sha256",
    "rfc6962_merkle_root", "rfc6962_inclusion_proof",
    # Crypto (3) — Phase 6A
    "ed25519_keypair_generate", "ed25519_sign", "ed25519_verify",
    # Serialization (1) — Phase 1
    "canonical_json",
    # Risk (1) — Phase 1
    "cvar",
    # Performance (2) — Phase 2
    "cagr",
    # Mean Reversion (2) — Phase 2
    "ornstein_uhlenbeck_fit", "ornstein_uhlenbeck_half_life",
    # Volatility (3) — Phase 2
    "garch_fit", "garch_forecast", "ewma_volatility",
    # Derivatives (3) — Phase 2
    "black_scholes_price", "black_scholes_greeks", "implied_volatility",
    # Derivatives (8) — Phase 5A
    "binomial_tree_price",
    "mc_european_price",
    "mc_asian_price",
    "barrier_option_price",
    "lookback_option_price",
    "lsm_american_price",
    "svensson_yield_curve",
    "cubic_spline_yield_curve",
    # Similarity (1) — Phase 3
    "vector_similarity",
    # Risk (Phase 4)
    "mean_deviation",
    # Timeseries (Phase 4)
    "adf_test",
    "kpss_test",
    "engle_granger_cointegration",
    "johansen_cointegration",
    "ljung_box_test",
    "durbin_watson",
    "granger_causality_test",
    "jarque_bera_test",
    "breusch_pagan_test",
    # Volatility (Phase 4)
    "egarch_fit",
    "egarch_forecast",
    "gjr_garch_fit",
    "gjr_garch_forecast",
    "realized_variance",
    "parkinson_volatility",
    "garman_klass_volatility",
    "yang_zhang_volatility",
    # Technical (Phase 4)
    "kama",
    "stochastic_oscillator",
    "cci",
    "williams_r",
    "keltner_channels",
    "obv",
    "mfi",
    # Phase 9A (v1.11.0)
    "path_signature_compute",
    "fisher_rao_distance",
    "rough_volatility_simulate",
    "sabr_implied_volatility",
    # Phase 10 Behavioral (v2.0.0)
    "cpt_value_function",
    "probability_weighting_function",
    "salience_function",
    "large_loss_aversion_degree",
    "salience_ranking_weights",
    # Phase 10 Spectral (v2.0.0)
    "marchenko_pastur_threshold",
    "rotationally_invariant_estimator",
    "ledoit_wolf_shrinkage",
    "spectral_eigengap_detect",
    # Phase 10 Recursive Utility (v2.0.0)
    "epstein_zin_aggregator",
]
