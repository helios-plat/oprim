"""Tests for signal_analysis oprims (14 new)."""

import pytest

from oprim.signal_analysis import (bayesian_factor_posterior, divergence_score, abstain_decision, correlation_matrix, signal_temporal_decay, signal_rarity_weight, trend_sentiment_synergy, cross_timeframe_consistency, signal_failure_audit, pack_promotion_test, ic_oos_decay, factor_attribution, regime_conditional_ic, cross_sectional_rank, SignalAnalysisError)


def test_bayesian_factor_posterior():
    r = bayesian_factor_posterior(prior={"mean": 0, "std": 1}, likelihood={"mean": 0.5, "std": 0.5})
    assert r["ci_low"] < r["posterior_mean"] < r["ci_high"]


def test_bayesian_factor_posterior_invalid():
    with pytest.raises(SignalAnalysisError):
        bayesian_factor_posterior(prior={"mean": 0, "std": 0}, likelihood={"mean": 1, "std": 1})


def test_divergence_score_js():
    r = divergence_score(signal_a=[1, 2, 3, 4, 5] * 10, signal_b=[1, 2, 3, 4, 5] * 10)
    assert r == pytest.approx(0, abs=0.01)


def test_divergence_score_different():
    r = divergence_score(signal_a=[1] * 50, signal_b=[10] * 50, method="wasserstein")
    assert r > 0


def test_divergence_score_empty():
    with pytest.raises(SignalAnalysisError):
        divergence_score(signal_a=[], signal_b=[1])


def test_abstain_decision_proceed():
    assert abstain_decision(confidence=0.9, uncertainty=0.1) == "proceed"


def test_abstain_decision_abstain():
    assert abstain_decision(confidence=0.3, uncertainty=0.8) == "abstain"


def test_correlation_matrix_perfect():
    r = correlation_matrix(returns={"A": [1, 2, 3, 4, 5], "B": [2, 4, 6, 8, 10]})
    assert r["A"]["B"] == pytest.approx(1.0, abs=0.001)


def test_signal_temporal_decay():
    assert signal_temporal_decay(signal=1.0, age_hours=24, half_life=24) == pytest.approx(0.5)


def test_signal_temporal_decay_zero_age():
    assert signal_temporal_decay(signal=1.0, age_hours=0, half_life=24) == 1.0


def test_signal_rarity_weight_idf():
    r = signal_rarity_weight(signal_frequency=1, total_signals=100)
    assert r > signal_rarity_weight(signal_frequency=50, total_signals=100)


def test_signal_rarity_weight_invalid():
    with pytest.raises(SignalAnalysisError):
        signal_rarity_weight(signal_frequency=0, total_signals=100)


def test_trend_sentiment_synergy_dampen():
    r = trend_sentiment_synergy(trend_signal=0.8, sentiment_score=0.9)
    assert abs(r) < 0.8  # dampened


def test_trend_sentiment_synergy_boost():
    r = trend_sentiment_synergy(trend_signal=0.8, sentiment_score=-0.5)
    assert abs(r) > 0.8  # boosted (contrarian)


def test_cross_timeframe_consistency_aligned():
    r = cross_timeframe_consistency(tf1_signal=0.8, tf4_signal=0.6)
    assert r > 0.5


def test_cross_timeframe_consistency_divergent():
    r = cross_timeframe_consistency(tf1_signal=0.8, tf4_signal=-0.6)
    assert abs(r) < 0.5  # penalized


def test_signal_failure_audit():
    r = signal_failure_audit(signal_id="s1", actual_outcome=-0.05, predicted_score=0.8)
    assert r["direction_correct"] is False
    assert r["error"] > 0


def test_pack_promotion_test_promote():
    r = pack_promotion_test(historical_packs=[{"score": 50}, {"score": 55}], new_pack={"score": 80})
    assert r["improvement"] > 0


def test_pack_promotion_test_empty_history():
    r = pack_promotion_test(historical_packs=[], new_pack={"score": 75})
    assert r["promote"] is True


def test_ic_oos_decay():
    r = ic_oos_decay(ic_series=[0.1, 0.09, 0.08, 0.05, 0.03], oos_start_idx=2)
    assert r["ic_decay_slope"] < 0


def test_ic_oos_decay_invalid():
    with pytest.raises(SignalAnalysisError):
        ic_oos_decay(ic_series=[], oos_start_idx=0)


def test_factor_attribution():
    r = factor_attribution(fusion_score=72, factor_contributions={"trend": 30, "flow": 20})
    assert sum(r["attributions"].values()) + r["residual"] == pytest.approx(1.0, abs=0.01)


def test_regime_conditional_ic():
    r = regime_conditional_ic(ic_series=[0.1, -0.05, 0.08], regime_labels=["bull", "bear", "bull"])
    assert r["bull"] > 0
    assert r["bear"] < 0


def test_regime_conditional_ic_mismatch():
    with pytest.raises(SignalAnalysisError):
        regime_conditional_ic(ic_series=[0.1], regime_labels=["a", "b"])


def test_cross_sectional_rank_percentile():
    r = cross_sectional_rank(asset_scores={"BTC": 80, "ETH": 60, "SOL": 40})
    assert r["BTC"] == 1.0
    assert r["SOL"] == 0.0


def test_cross_sectional_rank_zscore():
    r = cross_sectional_rank(asset_scores={"A": 100, "B": 50}, method="zscore")
    assert r["A"] > 0
    assert r["B"] < 0
