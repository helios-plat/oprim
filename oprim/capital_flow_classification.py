# 🚫 Tide-only — A股主力/超大/大/中/小单分类，永久 Tide 专属
"""A股主力资金流分类 (东方财富/Wind/Tushare 行业标准)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

_DEFAULT_THRESHOLDS: dict[str, int] = {
    "small": 40_000,
    "medium": 200_000,
    "large": 1_000_000,
    "extra_large": 10_000_000,  # sentinel, amounts >= large are extra_large
}

_BLOCK_TRADE_THRESHOLD = 5_000_000


def capital_flow_classification(
    trades: pd.DataFrame,
    *,
    tier_thresholds: dict[str, int] | None = None,
    direction_classification: Literal["price_tick", "lee_ready", "tick_volume"] = "lee_ready",
    market_value_at_time: float | None = None,
) -> dict[str, Any]:
    """A股主力/超大/大/中/小单分类 (东方财富/Wind/Tushare 行业标准).

    Args:
        trades:                    pd.DataFrame [price, volume, amount, ...] (已排序)
        tier_thresholds:           tier 阈值 dict (默认 A股惯例)
        direction_classification:  买卖方向判定算法
        market_value_at_time:      总市值 (提供则计算占比)

    Returns:
        dict with tier_labels, direction_labels, tier_stats, net_flow,
        optionally pct_of_market_value.

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
    volumes = trades["volume"].to_numpy(dtype=np.float64)
    amounts = trades["amount"].to_numpy(dtype=np.float64)

    # Classify tiers
    tier_labels = _classify_tiers(amounts, thresholds)

    # Classify directions
    direction_labels = _classify_directions(prices, direction_classification)

    # Compute tier statistics. block_trade (单笔 ≥500万) is the strongest 主力 print —
    # it must carry its own stats AND fold into 主力净流入, not be silently dropped.
    tier_stats: dict[str, dict[str, float]] = {}
    tiers = ["block_trade", "extra_large", "large", "medium", "small"]
    for tier in tiers:
        mask = tier_labels == tier
        buy_mask = mask & (direction_labels == "buy")
        sell_mask = mask & (direction_labels == "sell")
        tier_stats[tier] = {
            "count": int(mask.sum()),
            "total_volume": float(volumes[mask].sum()),
            "total_amount": float(amounts[mask].sum()),
            "net_buy_amount": float(amounts[buy_mask].sum() - amounts[sell_mask].sum()),
            "avg_amount": float(amounts[mask].mean()) if mask.sum() > 0 else 0.0,
            "buy_count": int(buy_mask.sum()),
            "sell_count": int(sell_mask.sum()),
        }

    # Net flows
    def _net(tier: str) -> float:
        return tier_stats[tier]["net_buy_amount"]

    # 主力净流入 = 超大单 + 大单 + 特大单(block_trade). Previously block_trade was
    # excluded, systematically under-counting the largest institutional prints.
    net_flow = {
        "main": _net("block_trade") + _net("extra_large") + _net("large"),
        "block_trade": _net("block_trade"),
        "extra_large": _net("extra_large"),
        "large": _net("large"),
        "medium": _net("medium"),
        "small": _net("small"),
    }

    result: dict[str, Any] = {
        "tier_labels": tier_labels,
        "direction_labels": direction_labels,
        "tier_stats": tier_stats,
        "net_flow": net_flow,
    }

    if market_value_at_time is not None:
        if market_value_at_time <= 0:
            raise ValueError("market_value_at_time must be > 0")
        result["pct_of_market_value"] = {k: v / market_value_at_time for k, v in net_flow.items()}

    return result


def _classify_tiers(amounts: np.ndarray, thresholds: dict[str, int]) -> np.ndarray:
    small_thresh = thresholds["small"]
    medium_thresh = thresholds["medium"]
    large_thresh = thresholds["large"]

    labels = np.full(len(amounts), "small", dtype=object)
    labels[amounts >= small_thresh] = "medium"
    labels[amounts >= medium_thresh] = "large"
    labels[amounts >= large_thresh] = "extra_large"
    # Block trades override
    labels[amounts >= _BLOCK_TRADE_THRESHOLD] = "block_trade"
    return labels


def _classify_directions(
    prices: np.ndarray,
    method: str,
) -> np.ndarray:
    n = len(prices)
    directions = np.full(n, "buy", dtype=object)

    if method == "price_tick":
        for i in range(1, n):
            if prices[i] < prices[i - 1]:
                directions[i] = "sell"
            elif prices[i] == prices[i - 1]:
                directions[i] = directions[i - 1]

    elif method == "lee_ready":
        # Lee-Ready (1991): compare to rolling median midpoint; fallback to tick test
        window = 10
        for i in range(n):
            start = max(0, i - window)
            mid = float(np.median(prices[start : i + 1]))
            if prices[i] > mid:
                directions[i] = "buy"
            elif prices[i] < mid:
                directions[i] = "sell"
            else:
                # tick test fallback
                if i > 0:
                    directions[i] = "buy" if prices[i] >= prices[i - 1] else "sell"

    elif method == "tick_volume":
        for i in range(1, n):
            directions[i] = "buy" if prices[i] >= prices[i - 1] else "sell"

    return directions


def _validate_thresholds(t: dict[str, int]) -> None:
    required = {"small", "medium", "large"}
    missing = required - set(t.keys())
    if missing:
        raise ValueError(f"tier_thresholds missing keys: {sorted(missing)}")
    if not (t["small"] < t["medium"] < t["large"]):
        raise ValueError("tier_thresholds must be strictly ascending: small < medium < large")
