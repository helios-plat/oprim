"""Pricing lookup atomic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def pricing_lookup(query: Mapping[str, Any], *, table: Any) -> Any:
    """Look up pricing in the injected obase pricing table."""
    if hasattr(table, "lookup"):
        return table.lookup(
            str(query.get("category", "llm")),
            str(query.get("provider", "")),
            str(query.get("model", query.get("model_or_tier", ""))),
            str(query.get("unit", "per_token")),
        )
    if isinstance(table, Mapping):
        key = (
            str(query.get("provider", "")),
            str(query.get("model", query.get("model_or_tier", ""))),
        )
        return table.get(key) or table.get("/".join(key))
    return None


__all__ = ["pricing_lookup"]
