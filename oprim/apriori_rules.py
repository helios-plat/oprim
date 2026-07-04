# ADR-061-CANDIDATE
"""Apriori 关联规则挖掘 (Agrawal-Srikant 1994 VLDB) — 反哺候选."""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any


def apriori_rules(
    transactions: list[list[str]] | list[set[str]],
    *,
    min_support: float = 0.05,
    min_confidence: float = 0.5,
    min_lift: float = 1.0,
    max_length: int = 4,
    return_itemsets: bool = False,
) -> dict[str, Any]:
    """Apriori 关联规则挖掘 (Agrawal-Srikant 1994 VLDB).

    Args:
        transactions: 交易列表，每个交易是 item 集合
        min_support:   最小支持度 (0 < x <= 1)
        min_confidence: 最小置信度
        min_lift:      最小提升度
        max_length:    最大 itemset 长度
        return_itemsets: 是否返回频繁 itemset

    Returns:
        dict with keys: rules, n_transactions, n_unique_items,
        optionally frequent_itemsets.

    Raises:
        ValueError: 参数不符合约束.
    """
    if not (0 < min_support <= 1):
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")
    if not (0 < min_confidence <= 1):
        raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}")
    if min_lift < 0:
        raise ValueError(f"min_lift must be >= 0, got {min_lift}")
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")
    if len(transactions) < 2:
        raise ValueError("Need at least 2 transactions")

    txn_sets: list[frozenset[str]] = [frozenset(t) for t in transactions]
    n_txns = len(txn_sets)

    all_items: set[str] = set()
    for t in txn_sets:
        all_items |= t
    n_unique = len(all_items)

    # Auto-reduce max_length for very large item sets
    if n_unique > 1000 and max_length > 3:
        warnings.warn(
            f"unique_items={n_unique} > 1000 with max_length={max_length}; "
            "降级到 max_length=3 防性能爆炸",
            RuntimeWarning,
            stacklevel=2,
        )
        max_length = 3

    # Count support: itemset → count
    def support_count(itemset: frozenset[str]) -> int:
        return sum(1 for t in txn_sets if itemset <= t)

    # Generate frequent 1-itemsets
    freq_itemsets: list[dict[str, Any]] = []
    freq_by_len: dict[int, list[frozenset[str]]] = {}

    freq_1: list[frozenset[str]] = []
    item_support: dict[frozenset[str], float] = {}
    for item in all_items:
        fs = frozenset([item])
        cnt = support_count(fs)
        sup = cnt / n_txns
        if sup >= min_support:
            freq_1.append(fs)
            item_support[fs] = sup
            freq_itemsets.append({"itemset": fs, "support": sup, "length": 1})

    freq_by_len[1] = freq_1
    current_freq = freq_1

    # Iterative candidate generation
    for k in range(2, max_length + 1):
        if not current_freq:
            break
        # Generate k-itemset candidates from (k-1)-itemsets
        candidates: set[frozenset[str]] = set()
        prev = current_freq
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                union = prev[i] | prev[j]
                if len(union) == k:
                    # Apriori pruning: all (k-1)-subsets must be frequent
                    prev_set = set(freq_by_len[k - 1])
                    if all(frozenset(c) in prev_set for c in combinations(union, k - 1)):
                        candidates.add(union)

        next_freq: list[frozenset[str]] = []
        for cand in candidates:
            cnt = support_count(cand)
            sup = cnt / n_txns
            if sup >= min_support:
                next_freq.append(cand)
                item_support[cand] = sup
                freq_itemsets.append({"itemset": cand, "support": sup, "length": k})

        freq_by_len[k] = next_freq
        current_freq = next_freq

    # Generate rules from all frequent itemsets (length >= 2)
    rules: list[dict[str, Any]] = []
    for entry in freq_itemsets:
        itemset = entry["itemset"]
        if len(itemset) < 2:
            continue
        sup_xy = entry["support"]
        # Generate all non-empty antecedent subsets
        items_list = list(itemset)
        for r in range(1, len(items_list)):
            for ant_items in combinations(items_list, r):
                ant = frozenset(ant_items)
                con = itemset - ant
                sup_x = item_support.get(ant)
                sup_y = item_support.get(con)
                if sup_x is None or sup_y is None:
                    continue
                conf = sup_xy / sup_x
                lift = conf / sup_y if sup_y > 0 else 0.0
                leverage = sup_xy - sup_x * sup_y
                if conf >= min_confidence and lift >= min_lift:
                    rules.append(
                        {
                            "antecedent": ant,
                            "consequent": con,
                            "support": sup_xy,
                            "confidence": conf,
                            "lift": lift,
                            "leverage": leverage,
                        }
                    )

    result: dict[str, Any] = {
        "rules": rules,
        "n_transactions": n_txns,
        "n_unique_items": n_unique,
    }
    if return_itemsets:
        result["frequent_itemsets"] = freq_itemsets
    return result
