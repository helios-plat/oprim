"""oprim._calculate_fitness_score — 单次适应度评分计算（时间衰减加权）.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: 纯数学计算（numpy），无外部依赖注入.

数学:
- Fitness = Σ(w_i * score_i), where w_i = exp(-λ * (t_now - t_i))
- 时间衰减因子 λ 控制"遗忘速度"
- 较新的奖励权重更高，避免被偶然历史高分绑架
"""

from __future__ import annotations

import math
import time
from typing import Any


def _calculate_fitness_score(
    reward_history: list[tuple[float, float]],
    *,
    decay_lambda: float = 0.1,
    min_weight: float = 0.01,
    recency_bias: float = 0.3,
) -> dict[str, Any]:
    """使用指数衰减对历史奖励进行加权求和，输出稳定适应度.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    公式:
        w_i = exp(-decay_lambda * hours_ago)
        fitness = Σ(w_i * score_i) / Σ(w_i)

    额外: recency_bias 给最近一次奖励额外的权重加成。

    Args:
        reward_history: [(timestamp, score), ...] 时间戳-奖励对列表
        decay_lambda: 衰减速率（越大遗忘越快）
        min_weight: 最小权重截断
        recency_bias: 最近奖励的额外加成比例

    Returns:
        {
            "fitness": float,
            "weighted_sum": float,
            "total_weight": float,
            "effective_samples": float,
            "decay_median_age_hours": float,
            "most_recent_score": float | None,
            "trend": "improving" | "declining" | "stable",
        }
    """
    if not reward_history:
        return {
            "fitness": 0.0,
            "weighted_sum": 0.0,
            "total_weight": 0.0,
            "effective_samples": 0.0,
            "decay_median_age_hours": 0.0,
            "most_recent_score": None,
            "trend": "stable",
        }

    now = time.time()
    weights: list[float] = []
    scores: list[float] = []
    ages_hours: list[float] = []

    for ts, score in reward_history:
        hours_ago = (now - ts) / 3600.0
        w = math.exp(-decay_lambda * hours_ago)
        if w < min_weight:
            w = min_weight
        weights.append(w)
        scores.append(score)
        ages_hours.append(hours_ago)

    # 最近奖励加成
    if len(weights) > 0:
        weights[-1] *= 1.0 + recency_bias

    total_weight = sum(weights)
    weighted_sum = sum(w * s for w, s in zip(weights, scores, strict=False))

    fitness = weighted_sum / total_weight if total_weight > 1e-15 else 0.0

    # 有效样本数 (ESS = (Σw)² / Σw²)
    sum_sq_weights = sum(w * w for w in weights)
    effective_samples = (total_weight * total_weight) / max(sum_sq_weights, 1e-15)

    # 衰减中位年龄
    sorted_ages = sorted(ages_hours)
    median_age = sorted_ages[len(sorted_ages) // 2] if sorted_ages else 0.0

    # 趋势分析: 最近 3 次 vs 之前 3 次
    trend = "stable"
    if len(scores) >= 4:
        recent_avg = sum(scores[-3:]) / 3
        older_avg = (
            sum(scores[:-3][-3:]) / 3
            if len(scores[:-3]) >= 3
            else sum(scores[:-3]) / max(len(scores[:-3]), 1)
        )
        if recent_avg > older_avg * 1.1:
            trend = "improving"
        elif recent_avg < older_avg * 0.9:
            trend = "declining"

    return {
        "fitness": round(fitness, 6),
        "weighted_sum": round(weighted_sum, 4),
        "total_weight": round(total_weight, 4),
        "effective_samples": round(effective_samples, 2),
        "decay_median_age_hours": round(median_age, 2),
        "most_recent_score": scores[-1] if scores else None,
        "trend": trend,
    }
