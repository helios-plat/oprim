"""oprim.ext_tax_calculate — calculate tax for a cart/order via a registered
obase.TaxProvider (category="tax").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import TaxOprimError


async def ext_tax_calculate(
    provider: str, *, address: dict[str, Any], items: list[Any]
) -> dict[str, Any]:
    """Calculate tax for a set of line items at a destination address.

    Args:
        provider: Provider name registered in ProviderRegistry (category="tax").
        address: Destination address (provider-specific shape).
        items: Line items to tax (provider-specific shape).

    Returns:
        Provider's calculate() response dict (typically includes `tax_cents`).

    Raises:
        TaxOprimError: Provider not registered or the provider call failed.
    """
    try:
        tax_provider = ProviderRegistry.get().generic("tax", provider)
    except ProviderNotFoundError as exc:
        raise TaxOprimError(f"tax provider not found: {provider!r}", cause=exc) from exc

    try:
        return await tax_provider.calculate(address=address, items=items)
    except Exception as exc:
        raise TaxOprimError(f"calculate failed: {exc}", cause=exc) from exc
