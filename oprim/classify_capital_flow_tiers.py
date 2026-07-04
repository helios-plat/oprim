"""主力/超大/大/中/小单 tier + Lee-Ready 买卖方向 — 单笔/逐笔分类, 不做聚合统计.

ASSUMPTION (flagged for review): unlike the other 3 Tide-service ports, no Tide
source file or call site names this function — it doesn't exist anywhere in
Tide today. ``capital_flow_classification`` (already ported) already computes
tier + direction labels internally as a step toward its aggregated tier_stats/
net_flow output; this is a thin facade over that same per-trade classification
step (reusing its threshold constants and direction classifier verbatim, not
duplicating them) for callers who want per-trade labels without the
aggregation. Confirm this interpretation matches the actual intended consumer
before relying on it.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from oprim.capital_flow_classification import (
    _DEFAULT_THRESHOLDS,
    _classify_directions,
    _classify_tiers,
    _validate_thresholds,
)


def classify_capital_flow_tiers(
    trades: pd.DataFrame,
    *,
    tier_thresholds: dict[str, int] | None = None,
    direction_classification: Literal["price_tick", "lee_ready", "tick_volume"] = "lee_ready",
) -> dict[str, Any]:
    """Classify each trade's capital-flow tier + buy/sell direction (per-trade).

    Reuses ``capital_flow_classification``'s tier thresholds and direction
    classifier — this is the per-trade labeling step alone, without the
    aggregated tier_stats/net_flow rollup that function also computes.

    Args:
        trades: pd.DataFrame [price, volume, amount, ...] (已排序).
        tier_thresholds: tier 阈值 dict (默认复用 capital_flow_classification 的 A 股惯例).
        direction_classification: 买卖方向判定算法.

    Returns:
        dict with tier_labels (np.ndarray), direction_labels (np.ndarray),
        thresholds (echoed, for traceability).

    Raises:
        ValueError: 缺少必需列或参数非法.
    """
    required_cols = {"price", "volume", "amount"}
    missing = required_cols - set(trades.columns)
    if missing:
        raise ValueError(f"trades DataFrame missing columns: {sorted(missing)}")

    if direction_classification not in ("price_tick", "lee_ready", "tick_volume"):
        raise ValueError(f"Invalid direction_classification: {direction_classification!r}")

    thresholds = tier_thresholds or _DEFAULT_THRESHOLDS
    _validate_thresholds(thresholds)

    prices = trades["price"].to_numpy(dtype=np.float64)
    amounts = trades["amount"].to_numpy(dtype=np.float64)

    tier_labels = _classify_tiers(amounts, thresholds)
    direction_labels = _classify_directions(prices, direction_classification)

    return {
        "tier_labels": tier_labels,
        "direction_labels": direction_labels,
        "thresholds": thresholds,
    }
