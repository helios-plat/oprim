"""_vlm_consensus 采样共识门测试 (sampler 注入, 零 token)。"""
import pytest

from oprim import (
    DEFAULT_CRITERIA_MIN,
    VARIANCE_SPREAD_MAX,
    VLM_CRITERIA,
    aggregate_vlm_samples,
    run_vlm_consensus,
    vlm_consensus_decision,
)


def _sample(score=0.95, claimed="cup"):
    return {c: score for c in VLM_CRITERIA} | {"claimedClass": claimed}


def test_aggregate_median_and_majority_class():
    samples = [_sample(0.9, "cup"), _sample(0.94, "cup"), _sample(0.92, "mug")]
    agg = aggregate_vlm_samples(samples)
    assert agg["claimedClass"] == "cup"
    assert abs(agg["scores"]["objectness"] - 0.92) < 1e-6
    assert agg["spread"]["objectness"] == pytest.approx(0.04, abs=1e-6)
    assert agg["sampleCount"] == 3


def test_decision_pass():
    agg = aggregate_vlm_samples([_sample(), _sample()])
    d = vlm_consensus_decision(agg)
    assert d["passed"] is True and d["verdict"] == "pass"


def test_decision_hard_failures_never_overridden():
    d = run_vlm_consensus(lambda i: _sample(), hard_failures=["silhouette IoU 0.5 below 0.8"])
    assert d["passed"] is False and d["verdict"] == "reject"
    assert "cannot grant" in " ".join(d["reasons"])


def test_decision_low_score_rejects():
    agg = aggregate_vlm_samples([_sample(0.5), _sample(0.55)])
    d = vlm_consensus_decision(agg)
    assert d["passed"] is False and d["verdict"] == "reject"
    assert any("below minimum" in r for r in d["reasons"])


def test_decision_high_spread_probes():
    # median >= min (0.835) 但跨样本 spread 高 (0.23) → 不确定 → probe
    samples = [_sample(0.72, "cup"), _sample(0.95, "mug")]
    agg = aggregate_vlm_samples(samples)
    d = vlm_consensus_decision(agg, claimed_class="cup")
    assert d["verdict"] == "probe"
    assert any("spread" in r for r in d["reasons"]) or any("contradicts" in r for r in d["reasons"])


def test_decision_class_contradiction_probes():
    agg = aggregate_vlm_samples([_sample(0.95, "gun"), _sample(0.95, "gun")])
    d = vlm_consensus_decision(agg, claimed_class="cup")
    assert d["verdict"] == "probe"
    assert any("contradicts" in r for r in d["reasons"])


def test_run_consensus_counts_samples():
    calls = []

    def sampler(i):
        calls.append(i)
        return _sample()

    d = run_vlm_consensus(sampler, n_samples=3)
    assert len(calls) == 3
    assert d["sampleCount"] == 3
    assert d["passed"] is True


def test_run_consensus_no_sampling_on_hard_failure():
    calls = []

    def sampler(i):
        calls.append(i)
        return _sample()

    d = run_vlm_consensus(sampler, n_samples=3, hard_failures=["x"])
    assert calls == []  # 硬失败时不邀请模型意见
    assert d["verdict"] == "reject"
