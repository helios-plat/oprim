"""Public facade for oprim.signal_analysis.

oprim 3.x moved the implementations into the private ``oprim._signal_analysis`` module
and, for most symbols, exposes them through thin public facade modules (see
``signal_processing.py`` / ``ic_oos_decay.py``). This aggregate facade was
missing after the 2.x→3.x refactor — callers doing ``from oprim.signal_analysis
import ...`` (oskill.signal_fusion_skills, helios analytics/collectors, and
oprim's own test_signal_* suite) hit ModuleNotFoundError.
Restored 2026-07-10 to re-export the full public surface of _signal_analysis.
"""

from oprim._signal_analysis import (
    SignalAnalysisError,
    bayesian_factor_posterior,
    divergence_score,
    abstain_decision,
    correlation_matrix,
    signal_temporal_decay,
    signal_rarity_weight,
    trend_sentiment_synergy,
    cross_timeframe_consistency,
    signal_failure_audit,
    pack_promotion_test,
    ic_oos_decay,
    factor_attribution,
    regime_conditional_ic,
    cross_sectional_rank,
)

__all__ = [
    "SignalAnalysisError",
    "bayesian_factor_posterior",
    "divergence_score",
    "abstain_decision",
    "correlation_matrix",
    "signal_temporal_decay",
    "signal_rarity_weight",
    "trend_sentiment_synergy",
    "cross_timeframe_consistency",
    "signal_failure_audit",
    "pack_promotion_test",
    "ic_oos_decay",
    "factor_attribution",
    "regime_conditional_ic",
    "cross_sectional_rank",
]
