"""Oprim — atomic operations library (Layer 1 meta-primitives)."""

from oprim.crypto_scoring import (
    CryptoScoringError,
    score_active_addresses_change,
    score_basis,
    score_cex_balance_change,
    score_etf_inflow,
    score_funding_rate,
    score_lth_change,
    score_ma200_position,
    score_ma50_slope,
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

from oprim.crypto_lookup import (
    CryptoLookupError,
    regime_score,
    seasonality_score,
    sector_rotation_score,
)

from oprim.crypto_technical import (
    CryptoTechnicalError,
    compute_cross_asset_divergence_revert,
    compute_stablecoin_event_revert,
    compute_vpvr,
    detect_pivots,
)

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

from oprim._caddy import (
    caddy_admin_post,
    caddy_admin_reload,
    caddy_certificates_status,
    caddy_routes_list,
)

from oprim.llm_judge_rerank import LLMCaller, RerankResult, llm_judge_rerank
from oprim.llm_query_expand import llm_query_expand
from oprim.parse_obsidian_tasks import ObsidianTask, parse_obsidian_tasks

# Aegis Batch 1 — infrastructure / ops primitives (v2.9.0)
from oprim._docker import (
    compose_down,
    compose_up,
    docker_container_inspect,
    docker_container_list,
    docker_container_logs,
    docker_container_restart,
    docker_container_start,
    docker_container_stats,
    docker_container_stop,
    docker_image_delete,
    docker_image_list,
    docker_image_pull,
    docker_network_list,
    docker_volume_delete,
    docker_volume_list,
)
from oprim._filesystem import (
    archive_to_targz,
    dir_archive_to_targz,
    disk_usage,
    file_checksum,
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
    tcp_port_check,
)
from oprim._postgres import (
    postgres_locks_status,
    postgres_pool_status,
    postgres_replication_lag,
    postgres_slow_queries,
    postgres_table_size,
)
from oprim._rabbitmq import (
    rabbitmq_connection_status,
    rabbitmq_consumer_status,
    rabbitmq_node_status,
    rabbitmq_queue_status,
)
from oprim._s3 import (
    s3_object_metadata,
    s3_upload_file,
)
from oprim._system import (
    cpu_memory_snapshot,
    process_list_top,
)
from oprim._version import __version__

# P7-B2 — Video Prompt Primitives + Frame Transition + Story Predict
from oprim.style_marker_prompt import StyleType, style_marker_prompt
from oprim.lighting_control_prompt import LightingType, lighting_control_prompt
from oprim.camera_motion_prompt import MotionType, camera_motion_prompt
from oprim.first_last_frame_transition import (
    FrameTransitionError,
    FrameTransitionProviderNotFoundError,
    first_last_frame_transition,
)
from oprim.video_edit_element_remove import (
    VideoEditError,
    VideoEditProviderNotFoundError,
    video_edit_element_remove,
)
from oprim.story_predict import (
    StoryPredictError,
    StoryPrediction,
    TimePrediction,
    story_predict,
)

# P6-B2 — Video Generation + Audience Analytics
from oprim.audience_feedback_extract import audience_feedback_extract
from oprim.audience_sentiment_analyze import audience_sentiment_analyze
from oprim.bilibili_comments_fetch import bilibili_comments_fetch
from oprim.bilibili_video_stats import bilibili_video_stats
from oprim.face_animation import face_animation
from oprim.image_to_video import image_to_video
from oprim.motion_prompt_translate import motion_prompt_translate
from oprim.video_quality_metrics import video_quality_metrics
from oprim.vlm_video_analyze import vlm_video_analyze
from oprim.youtube_comments_fetch import youtube_comments_fetch
from oprim.youtube_video_stats import youtube_video_stats

# Phase 10 additions (v2.0.0)
from oprim.behavioral import (
    cpt_value_function,
    large_loss_aversion_degree,
    probability_weighting_function,
    salience_function,
    salience_ranking_weights,
)

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
from oprim.timeseries.time_series_split import time_series_split
from oprim.timeseries.equity_curve_segment_label import equity_curve_segment_label
from oprim.timeseries.rolling_window_aggregate import rolling_window_aggregate
from oprim.markets.sector_strength_proxy import sector_strength_proxy
from oprim.stats.within_group_percentile import within_group_percentile
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
    # Aegis Batch 1 — PostgreSQL (v2.9.0)
    "postgres_pool_status",
    "postgres_slow_queries",
    "postgres_locks_status",
    "postgres_table_size",
    "postgres_replication_lag",
    # Aegis Batch 1 — RabbitMQ (v2.9.0)
    "rabbitmq_queue_status",
    "rabbitmq_connection_status",
    "rabbitmq_consumer_status",
    "rabbitmq_node_status",
    # Aegis Batch 1 — Caddy (v2.9.0)
    "caddy_admin_post",
    "caddy_admin_reload",
    "caddy_routes_list",
    "caddy_certificates_status",
    # Aegis Batch 1 — Network (v2.9.0)
    "tcp_port_check",
    "http_health_probe",
    "dns_resolve",
    "http_request_once",
    # Aegis Batch 1 — Filesystem (v2.9.0)
    "disk_usage",
    "archive_to_targz",
    "dir_archive_to_targz",
    "file_checksum",
    # Aegis Batch 1 — Metrics/Logs (v2.9.0)
    "prometheus_instant_query",
    "prometheus_range_query",
    "loki_log_query",
    "structlog_parse",
    # Aegis Batch 1 — System (v2.9.0)
    "cpu_memory_snapshot",
    "process_list_top",
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
]

# --- Tide v4 extraction: B1-B3 (11 oprims) ---
from oprim.kdj import kdj, KDJResult
from oprim.limit_status_calc import limit_status_calc, LimitStatusResult
from oprim.beneish_m_score import beneish_m_score, BeneishInput, BeneishResult
from oprim.dupont_decomposition import dupont_decomposition, DuPontResult
from oprim.dcf_valuation import dcf_valuation, DCFResult
from oprim.volume_ratio import volume_ratio
from oprim.apply_screen_filter import apply_screen_filter, ScreenRule, ScreenResult
from oprim.financial_metric_extraction import financial_metric_extraction, NewsItem, FinancialMetric
from oprim.policy_event_extraction import policy_event_extraction, PolicyNews, PolicyEvent
from oprim.industry_attribution import industry_attribution, IndustryImpact
from oprim.pattern_detection import pattern_detection, OHLCVInput, PatternMatch
from oprim.predicate import evaluate_threshold_condition, OperatorType

# --- Aegis Step 15 B2 — SSRF prevention (oprim 2.16.0) ---
from oprim.url_safety_check import URLSafetyError, URLSafetyResult, url_safety_check

# --- Aegis C2 B2 — webhook delivery (oprim 2.20.0) ---
from oprim.http_post_webhook import WebhookResult, http_post_webhook

# --- Aegis C2 B3 — threshold evaluator (oprim 2.20.0) ---
from oprim.evaluate_threshold_rule import (
    ThresholdResult,
    ThresholdRuleError,
    evaluate_threshold_rule,
)

# --- Aegis C2 B4 — throttle decision (oprim 2.20.0) ---
from oprim.should_throttle import should_throttle

# --- B9 — 7 realtime detector oprims (oprim 2.19.0) ---
from oprim._detector_types import DetectorSignal
from oprim.detect_sector_collapse import SectorCollapseConfig, detect_sector_collapse
from oprim.detect_dragon_switch import DragonSwitchConfig, detect_dragon_switch
from oprim.detect_hot_money_converge import HotMoneyConvergeConfig, detect_hot_money_converge
from oprim.detect_limit_board_explosion import (
    LimitBoardExplosionConfig,
    detect_limit_board_explosion,
)
from oprim.detect_volume_spike import VolumeSpikeConfig, detect_volume_spike
from oprim.detect_northbound_reversal import NorthboundReversalConfig, detect_northbound_reversal
from oprim.detect_news_shock import NewsShockConfig, detect_news_shock

# --- B8 — 13 utility/compute oprims (oprim 2.18.0) ---
from oprim.compute_seat_t3_return import SeatT3ReturnResult, compute_seat_t3_return
from oprim.fetch_themes_daily import ThemeEntry, ThemesFetchError, fetch_themes_daily
from oprim.theme_to_sw_industry_mapping import ThemeSWMapping, theme_to_sw_industry_mapping
from oprim.fetch_sector_returns import SectorReturn, SectorFetchError, fetch_sector_returns
from oprim.pe_ttm_lookback_safe import PETTMResult, pe_ttm_lookback_safe
from oprim.stop_loss_compliance_check import StopLossResult, stop_loss_compliance_check
from oprim.realtime_quote_redis_fetch import (
    QuoteResult,
    QuoteFetchError,
    realtime_quote_redis_fetch,
)
from oprim.stamp_tax_rate_by_date import StampTaxResult, stamp_tax_rate_by_date
from oprim.broker_export_render import BrokerExportResult, broker_export_render
from oprim.compliance_disclaimer_inject import compliance_disclaimer_inject
from oprim.monthly_review_jinja2_render import RenderedReport, monthly_review_jinja2_render
from oprim.train_val_oos_splitter import TrainValOOSSplit, train_val_oos_splitter
from oprim.detect_volume_dryup_breakout import VolumeBreakoutResult, detect_volume_dryup_breakout

# --- B7 — 8 macro data fetch oprims (oprim 2.17.0) ---
from oprim._macro_types import MacroDataPoint, MacroFetchError
from oprim.fetch_macro_m2 import fetch_macro_m2
from oprim.fetch_macro_pboc import fetch_macro_pboc
from oprim.fetch_macro_cpi_ppi_pmi import fetch_macro_cpi_ppi_pmi
from oprim.fetch_macro_lpr import fetch_macro_lpr
from oprim.fetch_macro_rrr import fetch_macro_rrr
from oprim.fetch_macro_yield_spread import fetch_macro_yield_spread
from oprim.fetch_macro_calendar import fetch_macro_calendar
from oprim.fetch_macro_policy_news import fetch_macro_policy_news
