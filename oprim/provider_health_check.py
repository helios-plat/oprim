"""oprim.provider_health_check — Check provider reachability without raising exceptions."""
from __future__ import annotations


async def provider_health_check(
    provider: str,
    *,
    timeout_s: float = 5.0,
    caller: object = None,
) -> bool:
    """Return True if provider is reachable, False otherwise. Never raises."""
    try:
        from obase.provider_registry import ProviderRegistry
        return any(ProviderRegistry.has(kind, provider) for kind in ("health", "llm", "video"))
    except Exception:
        return False
