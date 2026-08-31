"""Atomic lookup of one versioned native/MCP tool contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from obase.tool_governance import ToolSpec


def tool_resolve(identity: str, *, registry: Any) -> ToolSpec:
    """Resolve one tool identity through an injected registry."""
    value: Any
    if isinstance(registry, Mapping) or hasattr(registry, "get"):
        value = registry.get(identity)
    elif hasattr(registry, "resolve"):
        value = registry.resolve(identity)
    else:
        value = registry(identity)
    if value is None:
        raise KeyError(f"tool contract not found: {identity}")
    if isinstance(value, ToolSpec):
        return value
    if isinstance(value, Mapping):
        return ToolSpec(**dict(value))
    raise TypeError("tool registry returned an unsupported contract")


__all__ = ["tool_resolve"]
