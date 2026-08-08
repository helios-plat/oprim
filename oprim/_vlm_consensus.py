"""VLM 采样共识门 (VLM Consensus Gate) — 模型意见的受控注入原语。

从 img2threejs forge/stage4_review/vlm_gate.py 提炼的核心规则层:
模型 (VLM) 是"被门控、被校准、被交叉核对"的润滑剂, 而不是裁判。

规则 (与 forge Plan 1.3 §3.4 对齐):
1. VLM 永不运行在确定性硬门失败之上 (由调用方保证: 硬失败 → 直接 reject,
   不调用本层) — 几何损坏的渲染问模型只会得到"自信但错误"的 OK;
2. 多采样 self-consistency 投票: 每判据取 median, 跨样本 spread 高 → 不确定
   信号 → probe 而非掷硬币;
3. 声称类别与确定性几何描述符交叉核对 (模型语义不得覆盖测量几何);
4. 硬/软分治: VLM 不能授予硬几何失败通过 (它根本不会跑), 但可在
   确定性软性近阈值拒绝时, 在 criteria + evidence 一致的前提下挽救。

实际 VLM 调用以 sampler (callable) 注入 — 本层纯逻辑、可 stub 测试,
零 token 依赖。纯 stdlib。
"""

from __future__ import annotations

from typing import Any, Callable

VLM_CRITERIA: tuple[str, ...] = ("objectness", "semantic", "structural", "specular")
DEFAULT_CRITERIA_MIN: float = 0.80
VARIANCE_SPREAD_MAX: float = 0.20


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def aggregate_vlm_samples(
    samples: list[dict[str, Any]], criteria: tuple[str, ...] = VLM_CRITERIA
) -> dict[str, Any]:
    """聚合多采样 VLM 判定。

    Args:
        samples: 每次采样返回 {criterion: 0..1 score, claimedClass?: str, ...}。

    Returns:
        {scores: {crit: median}, spread: {crit: max-min}, claimedClass, sampleCount}
    """
    if not samples:
        return {"scores": {}, "spread": {}, "claimedClass": None, "sampleCount": 0}
    scores: dict[str, float] = {}
    spread: dict[str, float] = {}
    for crit in criteria:
        values = [float(s.get(crit, 0.0)) for s in samples if crit in s]
        if not values:
            continue
        scores[crit] = _median(values)
        spread[crit] = max(values) - min(values)
    classes = [s.get("claimedClass") for s in samples if s.get("claimedClass")]
    claimed = None
    if classes:
        counts: dict[str, int] = {}
        for cls in classes:
            counts[cls] = counts.get(cls, 0) + 1
        claimed = max(counts, key=counts.get)  # majority (ties → first seen)
    return {
        "scores": scores,
        "spread": spread,
        "claimedClass": claimed,
        "sampleCount": len(samples),
    }


def vlm_consensus_decision(
    agg: dict[str, Any],
    *,
    criteria: tuple[str, ...] = VLM_CRITERIA,
    criteria_min: float = DEFAULT_CRITERIA_MIN,
    spread_max: float = VARIANCE_SPREAD_MAX,
    hard_failures: list[str] | tuple[str, ...] = (),
    claimed_class: str | None = None,
) -> dict[str, Any]:
    """把聚合结果转成门判定。

    Rules:
    - hard_failures 非空 → verdict=reject (VLM 不得覆盖确定性硬失败);
    - 任一判据 median < criteria_min → reject;
    - spread > spread_max (高不确定) → verdict=probe (需更多证据/重采样);
    - claimed_class 与几何描述符矛盾 → probe (语义不得覆盖测量几何);
    - 全部达标 → pass。

    Returns:
        {passed, verdict, reasons, scores, spread, claimedClass}
    """
    reasons: list[str] = []
    if hard_failures:
        reasons.extend(f"deterministic hard failure: {f}" for f in hard_failures)
        reasons.append("VLM cannot grant past a deterministic hard failure")
        return {
            "passed": False,
            "verdict": "reject",
            "reasons": reasons,
            "scores": agg.get("scores", {}),
            "spread": agg.get("spread", {}),
            "claimedClass": agg.get("claimedClass"),
        }
    scores = agg.get("scores", {})
    spread = agg.get("spread", {})
    missing = [c for c in criteria if c not in scores]
    if missing:
        reasons.append(f"missing VLM criteria scores: {', '.join(missing)}")
        return {
            "passed": False,
            "verdict": "probe",
            "reasons": reasons,
            "scores": scores,
            "spread": spread,
            "claimedClass": agg.get("claimedClass"),
        }
    below = [c for c in criteria if scores[c] < criteria_min]
    if below:
        reasons.append(
            "criteria below minimum: " + ", ".join(f"{c}={scores[c]:.2f}" for c in below)
        )
    high_spread = [c for c in criteria if spread.get(c, 0.0) > spread_max]
    if high_spread:
        reasons.append(
            "high cross-sample spread (uncertain → probe): "
            + ", ".join(f"{c}={spread[c]:.2f}" for c in high_spread)
        )
    if claimed_class is not None and agg.get("claimedClass") and claimed_class != agg["claimedClass"]:
        reasons.append(
            f"claimed class '{agg['claimedClass']}' contradicts geometric descriptor '{claimed_class}'"
        )
    if below:
        return {
            "passed": False,
            "verdict": "reject",
            "reasons": reasons,
            "scores": scores,
            "spread": spread,
            "claimedClass": agg.get("claimedClass"),
        }
    if high_spread or any("contradicts" in r for r in reasons):
        return {
            "passed": False,
            "verdict": "probe",
            "reasons": reasons,
            "scores": scores,
            "spread": spread,
            "claimedClass": agg.get("claimedClass"),
        }
    return {
        "passed": True,
        "verdict": "pass",
        "reasons": ["all criteria at or above minimum with consistent samples"],
        "scores": scores,
        "spread": spread,
        "claimedClass": agg.get("claimedClass"),
    }


def run_vlm_consensus(
    sampler: Callable[[int], dict[str, Any]],
    *,
    n_samples: int = 3,
    criteria: tuple[str, ...] = VLM_CRITERIA,
    criteria_min: float = DEFAULT_CRITERIA_MIN,
    spread_max: float = VARIANCE_SPREAD_MAX,
    hard_failures: list[str] | tuple[str, ...] = (),
    claimed_class: str | None = None,
) -> dict[str, Any]:
    """执行完整采样共识: 采样 n 次 → 聚合 → 判定。

    sampler: callable(index) → 单次 VLM 判定 dict (含 criteria 分数)。
    调用方负责提供真实 VLM 的 prompt/视觉输入 (如 veya 视觉档或模型原生视觉)。
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if hard_failures:
        # 硬失败时连采样都不做 (模型意见不被邀请)
        return vlm_consensus_decision(
            {"scores": {}, "spread": {}, "claimedClass": None, "sampleCount": 0},
            criteria=criteria,
            criteria_min=criteria_min,
            spread_max=spread_max,
            hard_failures=hard_failures,
            claimed_class=claimed_class,
        )
    samples = [sampler(i) for i in range(n_samples)]
    agg = aggregate_vlm_samples(samples, criteria=criteria)
    decision = vlm_consensus_decision(
        agg,
        criteria=criteria,
        criteria_min=criteria_min,
        spread_max=spread_max,
        hard_failures=hard_failures,
        claimed_class=claimed_class,
    )
    decision["sampleCount"] = agg["sampleCount"]
    return decision
