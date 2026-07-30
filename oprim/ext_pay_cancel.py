"""oprim.ext_pay_cancel — cancel an authorized-but-not-captured payment
intent via a registered obase.PaymentProvider (category="payment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import PaymentOprimError


async def ext_pay_cancel(provider: str, *, intent_id: str) -> dict[str, Any]:
    """Cancel an authorized payment intent before it is captured.

    Args:
        provider: Provider name registered in ProviderRegistry (category="payment").
        intent_id: Intent ID of a previously authorized (not yet captured) payment.

    Returns:
        Provider's cancel() response dict.

    Raises:
        PaymentOprimError: Provider not registered or the provider call failed
            (e.g. intent already captured or refunded).
    """
    try:
        payment_provider = ProviderRegistry.get().generic("payment", provider)
    except ProviderNotFoundError as exc:
        raise PaymentOprimError(f"payment provider not found: {provider!r}", cause=exc) from exc

    try:
        return await payment_provider.cancel(intent_id=intent_id)
    except Exception as exc:
        raise PaymentOprimError(f"cancel failed: {exc}", cause=exc) from exc
