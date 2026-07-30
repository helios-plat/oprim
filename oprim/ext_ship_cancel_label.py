"""oprim.ext_ship_cancel_label — cancel a shipping label via a registered
obase.FulfillmentProvider (category="fulfillment").
"""

from __future__ import annotations

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import ShippingOprimError


async def ext_ship_cancel_label(provider: str, *, tracking_number: str) -> bool:
    """Cancel a previously created shipping label.

    Args:
        provider: Provider name registered in ProviderRegistry (category="fulfillment").
        tracking_number: Tracking number returned by a prior `ext_ship_create_label` call.

    Returns:
        True if the label was canceled, False if it was unknown or already canceled.

    Raises:
        ShippingOprimError: Provider not registered or the provider call itself raised.
    """
    try:
        fulfillment_provider = ProviderRegistry.get().generic("fulfillment", provider)
    except ProviderNotFoundError as exc:
        raise ShippingOprimError(
            f"fulfillment provider not found: {provider!r}", cause=exc
        ) from exc

    try:
        return await fulfillment_provider.cancel_label(tracking_number=tracking_number)
    except Exception as exc:
        raise ShippingOprimError(f"cancel_label failed: {exc}", cause=exc) from exc
