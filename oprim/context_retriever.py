"""3O 范式内化 — 上下文检索器 (oprim 确定性原语, 无 LLM).

n-gram 余弦相似 + 关键词规则打分, 纯 Python 零依赖。
"""

from __future__ import annotations

import math
import re
from typing import Any


def _ngrams(text: str, n: int = 3) -> dict[str, int]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    merged = " ".join(tokens)
    grams: dict[str, int] = {}
    for i in range(len(merged) - n + 1):
        gram = merged[i : i + n]
        grams[gram] = grams.get(gram, 0) + 1
    return grams


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_similar(query: str, *, corpus: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """单次原子检索: 与语料逐条计算 n-gram 余弦, 返回 top_k。

    Returns:
        [{index, text, score}] 按相似度降序。
    """
    qvec = _ngrams(query)
    scored = []
    for i, text in enumerate(corpus):
        score = _cosine(qvec, _ngrams(text))
        scored.append({"index": i, "text": text, "score": round(score, 4)})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored[:top_k] if s["score"] > 0]


def retrieve_rules(query: str, *, rules: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """单次原子检索: 关键词命中打分规则库。

    Returns:
        [{index, rule, score}] 按命中数降序。
    """
    q_words = set(w for w in re.findall(r"[a-z0-9_]+", query.lower()) if len(w) > 2)
    scored = []
    for i, rule in enumerate(rules):
        rule_words = set(re.findall(r"[a-z0-9_]+", rule.lower()))
        score = len(q_words & rule_words)
        scored.append({"index": i, "rule": rule, "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored[:top_k] if s["score"] > 0]
