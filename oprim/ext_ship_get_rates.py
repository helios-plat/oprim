"""oprim.ext_ship_get_rates — fetch shipping rate quotes via a registered
obase.FulfillmentProvider (category="fulfillment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import ShippingOprimError


async def ext_ship_get_rates(
    provider: str, *, package: dict[str, Any], address: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fetch shipping rate quotes for a package/address pair.

    Args:
        provider: Provider name registered in ProviderRegistry (category="fulfillment").
        package: Package dimensions/weight (provider-specific shape).
        address: Destination address (provider-specific shape).

    Returns:
        List of rate quote dicts (provider-specific shape, typically includes
        `carrier`/`service`/`rate_cents`).

    Raises:
        ShippingOprimError: Provider not registered or the provider call failed.
    """
    try:
        fulfillment_provider = ProviderRegistry.get().generic("fulfillment", provider)
    except ProviderNotFoundError as exc:
        raise ShippingOprimError(
            f"fulfillment provider not found: {provider!r}", cause=exc
        ) from exc

    try:
        return await fulfillment_provider.get_rates(package=package, address=address)
    except Exception as exc:
        raise ShippingOprimError(f"get_rates failed: {exc}", cause=exc) from exc
