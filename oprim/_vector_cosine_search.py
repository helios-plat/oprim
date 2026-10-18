"""oprim.vector_cosine_search — cosine similarity search over vectors (atomic)."""

from __future__ import annotations

import math
from typing import Any


def vector_cosine_search(
    *,
    query: list[float],
    corpus: list[dict[str, Any]],
    top_k: int = 5,
    vector_key: str = "vector",
) -> list[dict[str, Any]]:
    """Return top_k corpus items by cosine similarity to query.

    corpus items must contain vector_key. Returns items with added 'score'.
    """
    if not query:
        return []
    def cos(a, b):
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
    scored = []
    for item in corpus:
        v = item.get(vector_key)
        if not isinstance(v, (list, tuple)):
            continue
        scored.append({**item, "score": cos(query, list(v))})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
