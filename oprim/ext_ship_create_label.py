"""oprim.ext_ship_create_label — create a shipping label via a registered
obase.FulfillmentProvider (category="fulfillment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import ShippingOprimError


async def ext_ship_create_label(provider: str, *, shipment_info: dict[str, Any]) -> dict[str, Any]:
    """Create a shipping label for a shipment.

    Args:
        provider: Provider name registered in ProviderRegistry (category="fulfillment").
        shipment_info: Shipment details (provider-specific shape: items,
            addresses, chosen service, etc.).

    Returns:
        Provider's create_label() response dict (typically includes
        `tracking_number`).

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
        return await fulfillment_provider.create_label(shipment_info=shipment_info)
    except Exception as exc:
        raise ShippingOprimError(f"create_label failed: {exc}", cause=exc) from exc
