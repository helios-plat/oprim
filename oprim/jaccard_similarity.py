# ADR-061-CANDIDATE
"""Jaccard 相似度系数 — 距离/数据挖掘 (反哺候选)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def jaccard_similarity(
    set_a: np.ndarray | set[Any] | list[Any],
    set_b: np.ndarray | set[Any] | list[Any],
    *,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
    mode: Literal["binary", "weighted"] = "binary",
) -> float:
    """Jaccard 相似度系数 (Jaccard 1901, 加权扩展).

    Binary:   J(A,B) = |A ∩ B| / |A ∪ B|
    Weighted: J_w = sum_i min(w_a[i], w_b[i]) / sum_i max(w_a[i], w_b[i])

    Args:
        set_a, set_b: 元素集合
        weights_a, weights_b: 加权模式权重 (与 set_a/set_b 对齐)
        mode: "binary" 或 "weighted"

    Returns:
        float ∈ [0, 1]

    Raises:
        ValueError: 参数不符合约束.
    """
    if mode == "binary":
        a = set(set_a) if not isinstance(set_a, set) else set_a
        b = set(set_b) if not isinstance(set_b, set) else set_b
        if len(a) == 0 and len(b) == 0:
            return 1.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union

    # weighted mode
    a_arr = np.asarray(set_a)
    b_arr = np.asarray(set_b)
    if weights_a is None or weights_b is None:
        raise ValueError("weights_a and weights_b required for mode='weighted'")
    wa = np.asarray(weights_a, dtype=np.float64)
    wb = np.asarray(weights_b, dtype=np.float64)
    if np.any(wa < 0) or np.any(wb < 0):
        raise ValueError("weights must be non-negative")
    if len(a_arr) != len(wa) or len(b_arr) != len(wb):
        raise ValueError("weights must align with set elements")

    # build union of elements using hash-based approach
    a_dict = {v: w for v, w in zip(a_arr.tolist(), wa.tolist(), strict=False)}
    b_dict = {v: w for v, w in zip(b_arr.tolist(), wb.tolist(), strict=False)}
    all_keys = set(a_dict.keys()) | set(b_dict.keys())

    if len(all_keys) == 0:
        return 1.0

    mins_sum = 0.0
    maxs_sum = 0.0
    for k in all_keys:
        wa_k = a_dict.get(k, 0.0)
        wb_k = b_dict.get(k, 0.0)
        mins_sum += min(wa_k, wb_k)
        maxs_sum += max(wa_k, wb_k)

    if maxs_sum < 1e-12:
        return 1.0
    return mins_sum / maxs_sum
