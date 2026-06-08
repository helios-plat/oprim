"""Oprim — atomic operations library (Layer 1 meta-primitives)."""

from oprim._caddy import (
    caddy_admin_config,
    caddy_admin_post,
    caddy_admin_reload,
    caddy_admin_routes,
    caddy_certificates_status,
    caddy_route_add_atomic,
    caddy_route_remove_atomic,
    caddy_routes_list,
)

# Aegis Batch 1 — infrastructure / ops primitives (v2.9.0)
# Aegis Batch 2 — multi-host + lifecycle (oprim v2.32.0)
# Aegis Batch C (oprim v2.34.0)
from oprim._docker import (
    ContainerCreateResult,
    ContainerExecResult,
    ContainerRenameResult,
    NetworkCreateResult,
    NetworkDeleteResult,
    NodeInfo,
    PruneResult,
    VolumeCreateResult,
    compose_down,
    compose_up,
    docker_compose_down,
    docker_compose_pull,
    docker_compose_up,
    docker_container_create,
    docker_container_exec,
    docker_container_inspect,
    docker_container_list,
    docker_container_logs,
    docker_container_rename,
    docker_container_restart,
    docker_container_start,
    docker_container_stats,
    docker_container_stop,
    docker_image_delete,
    docker_image_list,
    docker_image_pull,
    docker_inspect,
    docker_logs,
    docker_network_create,
    docker_network_delete,
    docker_network_list,
    docker_node_info,
    docker_ps,
    docker_restart,
    docker_stats,
    docker_system_prune,
    docker_volume_create,
    docker_volume_delete,
    docker_volume_list,
)
from oprim._filesystem import (
    archive_to_targz,
    dir_archive_to_targz,
    disk_usage,
    file_checksum,
    fs_disk_usage,
    fs_inode_check,
)
from oprim._metrics_logs import (
    loki_log_query,
    prometheus_instant_query,
    prometheus_range_query,
    structlog_parse,
)
from oprim._network import (
    dns_resolve,
    http_health_probe,
    http_request_once,
    network_dns_resolve,
    network_http_health,
    network_port_check,
    tcp_port_check,
)
from oprim._postgres import (
    postgres_locks,
    postgres_locks_status,
    postgres_long_running_queries,
    postgres_pool_status,
    postgres_replication_lag,
    postgres_slow_queries,
    postgres_table_size,
)
from oprim._rabbitmq import (
    rabbitmq_connection_status,
    rabbitmq_consumer_count,
    rabbitmq_consumer_status,
    rabbitmq_node_status,
    rabbitmq_queue_depth,
    rabbitmq_queue_status,
)
from oprim._s3 import (
    s3_object_metadata,
    s3_upload_file,
)

# Aegis Batch 2 — SSH (oprim v2.33.0)
from oprim._ssh import (
    SshExecResult,
    SshPortCheckResult,
    SshUploadResult,
    ssh_exec,
    ssh_file_upload,
    ssh_port_forward_check,
)
from oprim._system import (
    cpu_memory_snapshot,
    process_list_top,
    system_cpu_usage,
    system_load_avg,
    system_ram_usage,
)
from oprim._version import __version__
from oprim.appstore_catalog_fetch import AppCatalogEntry, appstore_catalog_fetch

# P6-B2 — Video Generation + Audience Analytics
from oprim.audience_feedback_extract import audience_feedback_extract
from oprim.audience_sentiment_analyze import audience_sentiment_analyze

# Phase 10 additions (v2.0.0)
from oprim.behavioral import (
    cpt_value_function,
    large_loss_aversion_degree,
    probability_weighting_function,
    salience_function,
    salience_ranking_weights,
)
from oprim.bilibili_comments_fetch import bilibili_comments_fetch
from oprim.bilibili_video_stats import bilibili_video_stats
from oprim.camera_motion_prompt import MotionType, camera_motion_prompt

# Phase 6A additions (v1.9.0)
from oprim.crypto.ed25519 import (
    ed25519_keypair_generate,
    ed25519_sign,
    ed25519_verify,
    generate_keypair,
    load_private_key_pem,
    load_public_key_pem,
    save_keypair_pem,
    sign,
    verify,
)
from oprim.crypto.hashing import hmac_sha256, sha256_hash
from oprim.crypto.merkle import rfc6962_inclusion_proof, rfc6962_merkle_root
from oprim.crypto_lookup import (
    CryptoLookupError,
    regime_score,
    seasonality_score,
    sector_rotation_score,
)
from oprim.crypto_scoring import (
    CryptoScoringError,
    score_active_addresses_change,
    score_basis,
    score_cex_balance_change,
    score_etf_inflow,
    score_funding_rate,
    score_lth_change,
    score_ma50_slope,
    score_ma200_position,
    score_ma_arrangement,
    score_max_pain_distance,
    score_mvrv_zscore,
    score_oi_change,
    score_options_skew,
    score_resistance_distance,
    score_stablecoin_inflow,
    score_support_distance,
    score_vpvr_position,
)
from oprim.crypto_technical import (
    CryptoTechnicalError,
    compute_cross_asset_divergence_revert,
    compute_stablecoin_event_revert,
    compute_vpvr,
    detect_pivots,
)
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
from oprim.face_animation import face_animation
from oprim.finance import (
    beta_alpha_ols,
    drawdown_curve,
    futures_curve_shape,
    nelson_siegel_yield_curve,
    sharpe_ratio,
    value_at_risk,
)
from oprim.first_last_frame_transition import (
    FrameTransitionError,
    FrameTransitionProviderNotFoundError,
    first_last_frame_transition,
)

# v2.24.1 — export fix (tts / image_generate / image_understand were missing)
from oprim.image_generate import image_generate
from oprim.image_to_video import image_to_video
from oprim.image_understand import image_understand
from oprim.info_geometry.fisher_rao import fisher_rao_distance
from oprim.information import ordinal_pattern, phase_randomize, shannon_entropy
from oprim.io_fetch import (
    FetchError,
    fetch_btc_spy_corr,
    fetch_coingecko_history,
    fetch_crypto,
    fetch_current_price,
    fetch_decision_count,
    fetch_equity_series,
    fetch_prefs,
    fetch_regime,
    fetch_regime_crisis_flips,
    fetch_rss,
    fetch_stablecoin_mcap,
    fetch_strategy_trades,
    fetch_yahoo_history,
    fetch_yahoo_quote,
    get_active_event,
    get_etf_weight_modifier,
    get_previous_30d,
    get_regime_by_date,
    get_stablecoin_change_7d,
    get_symbol_funding_rate,
    get_symbol_oi_change_7d,
)
from oprim.io_write import (
    WriteError,
    clear_event,
    is_deduped,
    refresh_view,
    send_alert,
    store_news,
    upsert_canonical_metric,
    upsert_equity_series,
    upsert_events,
    write_event,
    write_rows,
)
from oprim.lighting_control_prompt import LightingType, lighting_control_prompt
from oprim.llm_judge_rerank import LLMCaller, RerankResult, llm_judge_rerank
from oprim.llm_query_expand import llm_query_expand
from oprim.markets.sector_strength_proxy import sector_strength_proxy
from oprim.mean_reversion.ornstein_uhlenbeck import (
    ornstein_uhlenbeck_fit,
    ornstein_uhlenbeck_half_life,
)
from oprim.motion_prompt_translate import motion_prompt_translate
from oprim.numerics import clip_with_warning, logsumexp_safe, softmax_safe
from oprim.parse_obsidian_tasks import ObsidianTask, parse_obsidian_tasks
from oprim.performance.annualization import cagr

# Phase 2 additions (v1.5.0)
# Note: cumulative_returns below shadows the time_series one;
# the time_series version remains accessible via oprim.time_series directly.
from oprim.performance.cumulative import cumulative_returns
from oprim.point_process import hawkes_nll
from oprim.recursive_utility import epstein_zin_aggregator
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

# Phase 3 additions (v1.6.0)
from oprim.similarity.vector import vector_similarity
from oprim.spectral import (
    ledoit_wolf_shrinkage,
    marchenko_pastur_threshold,
    rotationally_invariant_estimator,
    spectral_eigengap_detect,
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
from oprim.stats.within_group_percentile import within_group_percentile
from oprim.story_predict import (
    StoryPredictError,
    StoryPrediction,
    TimePrediction,
    story_predict,
)

# P7-B2 — Video Prompt Primitives + Frame Transition + Story Predict
from oprim.style_marker_prompt import StyleType, style_marker_prompt
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
from oprim.timeseries.equity_curve_segment_label import equity_curve_segment_label
from oprim.timeseries.heteroskedasticity import breusch_pagan_test
from oprim.timeseries.rolling_window_aggregate import rolling_window_aggregate
from oprim.timeseries.stationarity import adf_test, kpss_test
from oprim.timeseries.time_series_split import time_series_split
from oprim.topology import persistence_landscape, takens_embed
from oprim.tts_synthesize import tts_synthesize
from oprim.video_edit_element_remove import (
    VideoEditError,
    VideoEditProviderNotFoundError,
    video_edit_element_remove,
)
from oprim.video_quality_metrics import video_quality_metrics
from oprim.vlm_video_analyze import vlm_video_analyze
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
from oprim.youtube_comments_fetch import youtube_comments_fetch
from oprim.youtube_video_stats import youtube_video_stats

__all__ = [
    "__version__",
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
    # Statistics (12)
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
    # Distance (6)
    "wasserstein_distance",
    "dtw_distance",
    "cosine_similarity_batch",
    "euclidean_distance_matrix",
    "symmetric_kl_divergence",
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
    # Technical (8) — Phase 1
    "sma",
    "ema",
    "vwap",
    "macd",
    "rsi_normalized",
    "bollinger_bands",
    "donchian_channel",
    "chandelier_exit",
    # Crypto (4) — Phase 1
    "sha256_hash",
    "hmac_sha256",
    "rfc6962_merkle_root",
    "rfc6962_inclusion_proof",
    # Crypto (3→9) — Phase 6A + Phase 3 v2.1.0
    "ed25519_keypair_generate",
    "ed25519_sign",
    "ed25519_verify",
    "generate_keypair",
    "sign",
    "verify",
    "save_keypair_pem",
    "load_private_key_pem",
    "load_public_key_pem",
    # Serialization (1) — Phase 1
    "canonical_json",
    # Risk (1) — Phase 1
    "cvar",
    # Performance (2) — Phase 2
    "cagr",
    # Mean Reversion (2) — Phase 2
    "ornstein_uhlenbeck_fit",
    "ornstein_uhlenbeck_half_life",
    # Volatility (3) — Phase 2
    "garch_fit",
    "garch_forecast",
    "ewma_volatility",
    # Derivatives (3) — Phase 2
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
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
    "llm_judge_rerank",
    "RerankResult",
    "LLMCaller",
    "llm_query_expand",
    "parse_obsidian_tasks",
    "ObsidianTask",
    # Phase 10 Recursive Utility (v2.8.0)
    "epstein_zin_aggregator",
    # Aegis Batch 1 — Docker (v2.9.0)
    "compose_down",
    "compose_up",
    "docker_container_inspect",
    "docker_container_list",
    "docker_container_logs",
    "docker_container_start",
    "docker_container_stop",
    "docker_container_restart",
    "docker_image_delete",
    "docker_image_list",
    "docker_image_pull",
    "docker_container_stats",
    "docker_network_list",
    "docker_volume_delete",
    "docker_volume_list",
    # Aegis IMPL SPEC v1.0 — Docker short-names + compose_pull (v2.31.0)
    "docker_logs",
    "docker_ps",
    "docker_restart",
    "docker_stats",
    "docker_inspect",
    "docker_compose_up",
    "docker_compose_down",
    "docker_compose_pull",
    # Aegis Batch 1 — PostgreSQL (v2.9.0)
    "postgres_pool_status",
    "postgres_slow_queries",
    "postgres_locks_status",
    "postgres_table_size",
    "postgres_replication_lag",
    # Aegis IMPL SPEC v1.0 — PostgreSQL aliases (v2.31.0)
    "postgres_long_running_queries",
    "postgres_locks",
    # Aegis Batch 1 — RabbitMQ (v2.9.0)
    "rabbitmq_queue_status",
    "rabbitmq_connection_status",
    "rabbitmq_consumer_status",
    "rabbitmq_node_status",
    # Aegis IMPL SPEC v1.0 — RabbitMQ focused wrappers (v2.31.0)
    "rabbitmq_queue_depth",
    "rabbitmq_consumer_count",
    # Aegis Batch 1 — Caddy (v2.9.0)
    "caddy_admin_post",
    "caddy_admin_reload",
    "caddy_routes_list",
    "caddy_certificates_status",
    # Aegis IMPL SPEC v1.0 — Caddy new ops (v2.31.0)
    "caddy_admin_config",
    "caddy_admin_routes",
    "caddy_route_add_atomic",
    "caddy_route_remove_atomic",
    # Aegis Batch 1 — Network (v2.9.0)
    "tcp_port_check",
    "http_health_probe",
    "dns_resolve",
    "http_request_once",
    # Aegis IMPL SPEC v1.0 — Network aliases (v2.31.0)
    "network_port_check",
    "network_http_health",
    "network_dns_resolve",
    # Aegis Batch 1 — Filesystem (v2.9.0)
    "disk_usage",
    "archive_to_targz",
    "dir_archive_to_targz",
    "file_checksum",
    # Aegis IMPL SPEC v1.0 — Filesystem new ops (v2.31.0)
    "fs_disk_usage",
    "fs_inode_check",
    # Aegis Batch 1 — Metrics/Logs (v2.9.0)
    "prometheus_instant_query",
    "prometheus_range_query",
    "loki_log_query",
    "structlog_parse",
    # Aegis Batch 1 — System (v2.9.0)
    "cpu_memory_snapshot",
    "process_list_top",
    # Aegis IMPL SPEC v1.0 — System focused wrappers (v2.31.0)
    "system_cpu_usage",
    "system_ram_usage",
    "system_load_avg",
    # Aegis IMPL SPEC v1.0 — AppStore (v2.31.0)
    "appstore_catalog_fetch",
    "AppCatalogEntry",
    # Aegis Batch 1 — S3 (v2.9.0)
    "s3_upload_file",
    "s3_object_metadata",
    # Sprint 11 — Timeseries (v2.12.0)
    "time_series_split",
    "equity_curve_segment_label",
    # Sprint 12 — Markets + Stats (v2.13.0)
    "sector_strength_proxy",
    "within_group_percentile",
    # Sprint 14 — Timeseries (v2.14.0)
    "rolling_window_aggregate",
    # P6-B2 — Video Generation + Audience Analytics
    "image_to_video",
    "face_animation",
    "motion_prompt_translate",
    "audience_sentiment_analyze",
    "audience_feedback_extract",
    "youtube_video_stats",
    "youtube_comments_fetch",
    "bilibili_video_stats",
    "bilibili_comments_fetch",
    "video_quality_metrics",
    "vlm_video_analyze",
    # v2.24.1 — export fix
    "image_generate",
    "image_understand",
    "tts_synthesize",
    # P7-B2 — Video Prompt Primitives + Frame Transition + Story Predict
    "style_marker_prompt",
    "StyleType",
    "lighting_control_prompt",
    "LightingType",
    "camera_motion_prompt",
    "MotionType",
    "first_last_frame_transition",
    "FrameTransitionError",
    "FrameTransitionProviderNotFoundError",
    "video_edit_element_remove",
    "VideoEditError",
    "VideoEditProviderNotFoundError",
    "story_predict",
    "StoryPrediction",
    "TimePrediction",
    "StoryPredictError",
    # Helios Wave 01 — Crypto Scoring (17 oprim)
    "score_ma200_position",
    "score_ma50_slope",
    "score_ma_arrangement",
    "score_stablecoin_inflow",
    "score_etf_inflow",
    "score_cex_balance_change",
    "score_funding_rate",
    "score_basis",
    "score_mvrv_zscore",
    "score_active_addresses_change",
    "score_lth_change",
    "score_options_skew",
    "score_max_pain_distance",
    "score_oi_change",
    "score_resistance_distance",
    "score_support_distance",
    "score_vpvr_position",
    "CryptoScoringError",
    # --- Helios Wave 01: Crypto Lookup (3) ---
    "regime_score",
    "seasonality_score",
    "sector_rotation_score",
    "CryptoLookupError",
    # --- Helios Wave 01: Crypto Technical (4) ---
    "compute_vpvr",
    "detect_pivots",
    "compute_cross_asset_divergence_revert",
    "compute_stablecoin_event_revert",
    "CryptoTechnicalError",
    # --- Helios Wave 01: IO Fetch (22) ---
    "FetchError",
    "fetch_yahoo_history",
    "fetch_yahoo_quote",
    "fetch_crypto",
    "fetch_stablecoin_mcap",
    "fetch_rss",
    "fetch_current_price",
    "fetch_coingecko_history",
    "fetch_equity_series",
    "fetch_decision_count",
    "fetch_strategy_trades",
    "fetch_prefs",
    "fetch_btc_spy_corr",
    "fetch_regime_crisis_flips",
    "fetch_regime",
    "get_active_event",
    "get_regime_by_date",
    "get_previous_30d",
    "get_stablecoin_change_7d",
    "get_etf_weight_modifier",
    "get_symbol_funding_rate",
    "get_symbol_oi_change_7d",
    # --- Helios Wave 01: IO Write (11) ---
    "WriteError",
    "write_rows",
    "store_news",
    "upsert_events",
    "upsert_canonical_metric",
    "upsert_equity_series",
    "write_event",
    "clear_event",
    "is_deduped",
    "refresh_view",
    "send_alert",
    # --- Aegis C2 B2 — webhook delivery (oprim 2.20.0) ---
    "WebhookResult",
    "http_post_webhook",
    # --- Aegis C2 B3 — threshold evaluator (oprim 2.20.0) ---
    "ThresholdResult",
    "ThresholdRuleError",
    "evaluate_threshold_rule",
    # --- Aegis C2 B4 — throttle decision (oprim 2.20.0) ---
    "should_throttle",
    # --- Aegis C2 B5 — dedup key (oprim 2.20.0) ---
    "compute_dedup_key",
    # --- Aegis Step 15 B2 — SSRF prevention ---
    "url_safety_check",
    "URLSafetyResult",
    "URLSafetyError",
    # --- Tide v4 extraction: B1-B3 (11 oprims) ---
    "kdj",
    "KDJResult",
    "limit_status_calc",
    "LimitStatusResult",
    "beneish_m_score",
    "BeneishInput",
    "BeneishResult",
    "dupont_decomposition",
    "DuPontResult",
    "dcf_valuation",
    "DCFResult",
    "volume_ratio",
    "apply_screen_filter",
    "ScreenRule",
    "ScreenResult",
    "financial_metric_extraction",
    "NewsItem",
    "FinancialMetric",
    "policy_event_extraction",
    "PolicyNews",
    "PolicyEvent",
    "industry_attribution",
    "IndustryImpact",
    "pattern_detection",
    "OHLCVInput",
    "PatternMatch",
    "evaluate_threshold_condition",
    "OperatorType",
    # --- B7 — 8 macro data fetch oprims ---
    "MacroDataPoint",
    "MacroFetchError",
    "fetch_macro_m2",
    "fetch_macro_pboc",
    "fetch_macro_cpi_ppi_pmi",
    "fetch_macro_lpr",
    "fetch_macro_rrr",
    "fetch_macro_yield_spread",
    "fetch_macro_calendar",
    "fetch_macro_policy_news",
    # --- B8 — 13 utility/compute oprims ---
    "SeatT3ReturnResult",
    "compute_seat_t3_return",
    "ThemeEntry",
    "ThemesFetchError",
    "fetch_themes_daily",
    "ThemeSWMapping",
    "theme_to_sw_industry_mapping",
    "SectorReturn",
    "SectorFetchError",
    "fetch_sector_returns",
    "PETTMResult",
    "pe_ttm_lookback_safe",
    "StopLossResult",
    "stop_loss_compliance_check",
    "QuoteResult",
    "QuoteFetchError",
    "realtime_quote_redis_fetch",
    "StampTaxResult",
    "stamp_tax_rate_by_date",
    "BrokerExportResult",
    "broker_export_render",
    "compliance_disclaimer_inject",
    "RenderedReport",
    "monthly_review_jinja2_render",
    "TrainValOOSSplit",
    "train_val_oos_splitter",
    "VolumeBreakoutResult",
    "detect_volume_dryup_breakout",
    # Exceptions
    "OprimAuthError",
    "OprimConnectionError",
    "OprimError",
    "OprimNotFoundError",
    "OprimTimeoutError",
    "OprimValidationError",
    # --- B9 — 7 realtime detector oprims ---
    "DetectorSignal",
    "SectorCollapseConfig",
    "detect_sector_collapse",
    "DragonSwitchConfig",
    "detect_dragon_switch",
    "HotMoneyConvergeConfig",
    "detect_hot_money_converge",
    "LimitBoardExplosionConfig",
    "detect_limit_board_explosion",
    "VolumeSpikeConfig",
    "detect_volume_spike",
    "NorthboundReversalConfig",
    "detect_northbound_reversal",
    "NewsShockConfig",
    "detect_news_shock",
    # Step-12 — markets-related (oprim 2.22.0)
    "detect_daily_limit_up",
    "detect_daily_limit_down",
    "t_plus_n_blocked",
    "compute_commission",
    "compute_stamp_tax",
    # Stratum B1 P0 (oprim 2.23.0)
    "template_render",
    "crypto_token_generate",
    "SizeLimitResult",
    "file_size_limiter",
    "FileTypeInfo",
    "file_type_detector",
    "HTTPResponse",
    "http_post",
    "db_insert",
    "db_query",
    "WriteResult",
    "db_write",
    "db_read",
    "db_soft_delete",
    "db_update",
    "MigrationResult",
    "migration_runner",
    "ParsedDocument",
    "ParsedMarkdown",
    "ParsedPlaintext",
    "DocumentStructure",
    "Page",
    "Table",
    "ImageRef",
    "Section",
    "file_parser_pdf",
    "file_parser_epub",
    "file_parser_html",
    "file_parser_markdown",
    "file_parser_plaintext",
    "document_structure_extractor",
    "SummarizeResult",
    "llm_summarize",
    "cache_invalidate",
    "UploadResult",
    "file_upload_handler",
    "TempFileResult",
    "temp_file_manager",
    "EmailResult",
    "push_email",
    "OTPResult",
    "otp_generate",
    "otp_verify",
    # Aegis C3-C4 (oprim 2.24.0)
    "compute_event_fingerprint",
    # AII 3O Batch 3a (oprim 2.25.0)
    "coherence_compute",
    "INDEPENDENT_SOURCES",
    "GRADE_LADDER",
    "entity_graph_search",
    "vector_encode",
    # AII 3O Batch 3b — supporting oprim
    "bm25_search",
    # AII 3O Batch 4a — P2 knowledge layer (oprim 2.26.0)
    "structural_chunk",
    "ku_gate_validate",
    "REASONING_TYPES",
    "VALID_KNOWLEDGE_TYPES",
    "VALID_GRADES",
    "llm_extract_ku",
    "llm_distill_strategy",
    # AII 3O Batch 5a — P3 Q-matrix (oprim 2.27.0)
    "build_q_matrix",
    # AII 3O Batch 5b — P5 causal + backtest (oprim 2.28.0)
    "cmi_verify",
    "backtest_stat",
    # B2a — feed/content group (oprim 2.29.0)
    "url_fetch_ssrf_safe",
    "fetch_rss_feed",
    "parse_atom_feed",
    "detect_feed_url",
    "podcast_episode_parser",
    "feed_diff_detector",
    "ocr_detect_text",
    # B2b — utility group (oprim 2.29.0)
    "concept_extractor",
    "keyword_alert_checker",
    "citation_formatter",
    "timeline_aggregator",
    "backlink_resolver",
    "graph_traversal",
    # B2 — search (oprim 2.30.0)
    "searxng_search",
    # Aegis Batch 2 — multi-host + lifecycle (v2.32.0)
    "ContainerCreateResult",
    "docker_container_create",
    "PruneResult",
    "docker_system_prune",
    "NodeInfo",
    "docker_node_info",
    # Aegis Batch C (oprim v2.34.0)
    "ContainerRenameResult",
    "docker_container_rename",
    # Aegis Batch D — Docker 补全 (oprim v2.35.0)
    "ContainerExecResult",
    "docker_container_exec",
    "NetworkCreateResult",
    "docker_network_create",
    "NetworkDeleteResult",
    "docker_network_delete",
    "VolumeCreateResult",
    "docker_volume_create",
    # Aegis Batch 2 — SSH (v2.33.0)
    "SshExecResult",
    "ssh_exec",
    "SshUploadResult",
    "ssh_file_upload",
    "SshPortCheckResult",
    "ssh_port_forward_check",
]

# --- Tide v4 extraction: B1-B3 (11 oprims) ---
# --- B9 — 7 realtime detector oprims (oprim 2.19.0) ---
from oprim._detector_types import DetectorSignal
from oprim._document_types import (
    DocumentStructure,
    ImageRef,
    Page,
    ParsedDocument,
    ParsedMarkdown,
    ParsedPlaintext,
    Section,
    Table,
)
from oprim._exceptions import (
    OprimAuthError,
    OprimConnectionError,
    OprimError,
    OprimNotFoundError,
    OprimTimeoutError,
    OprimValidationError,
)

# --- B7 — 8 macro data fetch oprims (oprim 2.17.0) ---
from oprim._macro_types import MacroDataPoint, MacroFetchError
from oprim.apply_screen_filter import ScreenResult, ScreenRule, apply_screen_filter
from oprim.backlink_resolver import backlink_resolver
from oprim.backtest_stat import backtest_stat
from oprim.beneish_m_score import BeneishInput, BeneishResult, beneish_m_score

# --- AII 3O Batch 3b — supporting oprim (oprim 2.25.0) ---
from oprim.bm25_search import bm25_search
from oprim.broker_export_render import BrokerExportResult, broker_export_render

# --- AII 3O Batch 5a — P3 Q-matrix (oprim 2.27.0) ---
from oprim.build_q_matrix import build_q_matrix
from oprim.cache_invalidate import cache_invalidate
from oprim.citation_formatter import citation_formatter

# --- AII 3O Batch 5b — P5 causal + backtest (oprim 2.28.0) ---
from oprim.cmi_verify import cmi_verify

# --- AII 3O Batch 3a (oprim 2.25.0) ---
from oprim.coherence_compute import (
    GRADE_LADDER,
    INDEPENDENT_SOURCES,
    coherence_compute,
)
from oprim.compliance_disclaimer_inject import compliance_disclaimer_inject
from oprim.compute_commission import compute_commission

# --- Aegis C2 B5 — dedup key (oprim 2.20.0) ---
from oprim.compute_dedup_key import compute_dedup_key

# --- Aegis C3-C4 — error aggregation (oprim 2.24.0) ---
from oprim.compute_event_fingerprint import compute_event_fingerprint

# --- B8 — 13 utility/compute oprims (oprim 2.18.0) ---
from oprim.compute_seat_t3_return import SeatT3ReturnResult, compute_seat_t3_return
from oprim.compute_stamp_tax import compute_stamp_tax

# --- B2b — utility group (oprim 2.29.0) ---
from oprim.concept_extractor import concept_extractor
from oprim.crypto_token_generate import crypto_token_generate
from oprim.db_insert import db_insert
from oprim.db_query import db_query
from oprim.db_read import db_read
from oprim.db_soft_delete import db_soft_delete
from oprim.db_update import db_update
from oprim.db_write import WriteResult, db_write
from oprim.dcf_valuation import DCFResult, dcf_valuation
from oprim.detect_daily_limit_down import detect_daily_limit_down

# --- Step-12 — 5 markets-related oprims (oprim 2.22.0) ---
from oprim.detect_daily_limit_up import detect_daily_limit_up
from oprim.detect_dragon_switch import DragonSwitchConfig, detect_dragon_switch
from oprim.detect_feed_url import detect_feed_url
from oprim.detect_hot_money_converge import HotMoneyConvergeConfig, detect_hot_money_converge
from oprim.detect_limit_board_explosion import (
    LimitBoardExplosionConfig,
    detect_limit_board_explosion,
)
from oprim.detect_news_shock import NewsShockConfig, detect_news_shock
from oprim.detect_northbound_reversal import NorthboundReversalConfig, detect_northbound_reversal
from oprim.detect_sector_collapse import SectorCollapseConfig, detect_sector_collapse
from oprim.detect_volume_dryup_breakout import VolumeBreakoutResult, detect_volume_dryup_breakout
from oprim.detect_volume_spike import VolumeSpikeConfig, detect_volume_spike
from oprim.document_structure_extractor import document_structure_extractor
from oprim.dupont_decomposition import DuPontResult, dupont_decomposition
from oprim.entity_graph_search import entity_graph_search

# --- Aegis C2 B3 — threshold evaluator (oprim 2.20.0) ---
from oprim.evaluate_threshold_rule import (
    ThresholdResult,
    ThresholdRuleError,
    evaluate_threshold_rule,
)
from oprim.feed_diff_detector import feed_diff_detector
from oprim.fetch_macro_calendar import fetch_macro_calendar
from oprim.fetch_macro_cpi_ppi_pmi import fetch_macro_cpi_ppi_pmi
from oprim.fetch_macro_lpr import fetch_macro_lpr
from oprim.fetch_macro_m2 import fetch_macro_m2
from oprim.fetch_macro_pboc import fetch_macro_pboc
from oprim.fetch_macro_policy_news import fetch_macro_policy_news
from oprim.fetch_macro_rrr import fetch_macro_rrr
from oprim.fetch_macro_yield_spread import fetch_macro_yield_spread
from oprim.fetch_rss_feed import fetch_rss_feed
from oprim.fetch_sector_returns import SectorFetchError, SectorReturn, fetch_sector_returns
from oprim.fetch_themes_daily import ThemeEntry, ThemesFetchError, fetch_themes_daily
from oprim.file_parser_epub import file_parser_epub
from oprim.file_parser_html import file_parser_html
from oprim.file_parser_markdown import file_parser_markdown
from oprim.file_parser_pdf import file_parser_pdf
from oprim.file_parser_plaintext import file_parser_plaintext
from oprim.file_size_limiter import SizeLimitResult, file_size_limiter
from oprim.file_type_detector import FileTypeInfo, file_type_detector
from oprim.file_upload_handler import UploadResult, file_upload_handler
from oprim.financial_metric_extraction import FinancialMetric, NewsItem, financial_metric_extraction
from oprim.graph_traversal import graph_traversal
from oprim.http_post import HTTPResponse, http_post

# --- Aegis C2 B2 — webhook delivery (oprim 2.20.0) ---
from oprim.http_post_webhook import WebhookResult, http_post_webhook
from oprim.industry_attribution import IndustryImpact, industry_attribution
from oprim.kdj import KDJResult, kdj
from oprim.keyword_alert_checker import keyword_alert_checker
from oprim.ku_gate_validate import (
    REASONING_TYPES,
    VALID_GRADES,
    VALID_KNOWLEDGE_TYPES,
    ku_gate_validate,
)
from oprim.limit_status_calc import LimitStatusResult, limit_status_calc
from oprim.llm_distill_strategy import llm_distill_strategy
from oprim.llm_extract_ku import llm_extract_ku
from oprim.llm_summarize import SummarizeResult, llm_summarize
from oprim.migration_runner import MigrationResult, migration_runner
from oprim.monthly_review_jinja2_render import RenderedReport, monthly_review_jinja2_render
from oprim.ocr_detect_text import ocr_detect_text
from oprim.otp_generate import OTPResult, otp_generate, otp_verify
from oprim.parse_atom_feed import parse_atom_feed
from oprim.pattern_detection import OHLCVInput, PatternMatch, pattern_detection
from oprim.pe_ttm_lookback_safe import PETTMResult, pe_ttm_lookback_safe
from oprim.podcast_episode_parser import podcast_episode_parser
from oprim.policy_event_extraction import PolicyEvent, PolicyNews, policy_event_extraction
from oprim.predicate import OperatorType, evaluate_threshold_condition
from oprim.push_email import EmailResult, push_email
from oprim.realtime_quote_redis_fetch import (
    QuoteFetchError,
    QuoteResult,
    realtime_quote_redis_fetch,
)

# --- B2 — search (oprim 2.30.0) ---
from oprim.searxng_search import searxng_search

# --- Aegis C2 B4 — throttle decision (oprim 2.20.0) ---
from oprim.should_throttle import should_throttle
from oprim.stamp_tax_rate_by_date import StampTaxResult, stamp_tax_rate_by_date
from oprim.stop_loss_compliance_check import StopLossResult, stop_loss_compliance_check

# --- AII 3O Batch 4a — P2 knowledge layer (oprim 2.26.0) ---
from oprim.structural_chunk import structural_chunk
from oprim.t_plus_n_blocked import t_plus_n_blocked
from oprim.temp_file_manager import TempFileResult, temp_file_manager

# --- Stratum B1 P0 — 24 elements (oprim 2.23.0) ---
from oprim.template_render import template_render
from oprim.theme_to_sw_industry_mapping import ThemeSWMapping, theme_to_sw_industry_mapping
from oprim.timeline_aggregator import timeline_aggregator
from oprim.train_val_oos_splitter import TrainValOOSSplit, train_val_oos_splitter

# --- B2a — feed/content group (oprim 2.29.0) ---
from oprim.url_fetch_ssrf_safe import url_fetch_ssrf_safe

# --- Aegis Step 15 B2 — SSRF prevention (oprim 2.16.0) ---
from oprim.url_safety_check import URLSafetyError, URLSafetyResult, url_safety_check
from oprim.vector_encode import vector_encode
from oprim.volume_ratio import volume_ratio
