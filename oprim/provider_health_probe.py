"""Atomic provider health probe."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from typing import Any


async def provider_health_probe(
    provider: Any,
    *,
    probe: Callable[[Any], Any],
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Run one injected health probe for one provider."""
    name = (
        str(provider.get("name", ""))
        if isinstance(provider, Mapping)
        else str(getattr(provider, "name", provider))
    )
    started = time.monotonic()
    try:
        result = probe(provider)
        if inspect.isawaitable(result):
            import asyncio

            result = await asyncio.wait_for(result, timeout=timeout_s)
        elapsed = (time.monotonic() - started) * 1000.0
        if isinstance(result, Mapping):
            healthy = bool(result.get("healthy", result.get("ok", False)))
            return {
                "provider": name,
                "healthy": healthy,
                "status": str(result.get("status", "healthy" if healthy else "error")),
                "latency_ms": round(float(result.get("latency_ms", elapsed)), 2),
                "reason": str(result.get("reason", "")),
            }
        healthy = bool(result)
        return {
            "provider": name,
            "healthy": healthy,
            "status": "healthy" if healthy else "error",
            "latency_ms": round(elapsed, 2),
            "reason": "" if healthy else "probe returned false",
        }
    except Exception as exc:  # noqa: BLE001 - normalize the atomic boundary
        return {
            "provider": name,
            "healthy": False,
            "status": "error",
            "latency_ms": round((time.monotonic() - started) * 1000.0, 2),
            "reason": type(exc).__name__,
        }


__all__ = ["provider_health_probe"]
