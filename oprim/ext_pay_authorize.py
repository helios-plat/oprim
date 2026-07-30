"""oprim.ext_pay_authorize — authorize a payment via a registered
obase.PaymentProvider (category="payment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import PaymentOprimError


async def ext_pay_authorize(
    provider: str, *, amount: int, currency: str, meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Authorize a payment intent through the named payment provider.

    Args:
        provider: Provider name registered in ProviderRegistry (category="payment").
        amount: Amount in minor currency units (e.g. cents).
        currency: ISO currency code (e.g. "USD", "CNY").
        meta: Optional provider-specific metadata.

    Returns:
        Provider's authorize() response dict (typically includes `intent_id`).

    Raises:
        PaymentOprimError: Provider not registered or the provider call failed.
    """
    try:
        payment_provider = ProviderRegistry.get().generic("payment", provider)
    except ProviderNotFoundError as exc:
        raise PaymentOprimError(f"payment provider not found: {provider!r}", cause=exc) from exc

    try:
        return await payment_provider.authorize(amount=amount, currency=currency, meta=meta)
    except Exception as exc:
        raise PaymentOprimError(f"authorize failed: {exc}", cause=exc) from exc
