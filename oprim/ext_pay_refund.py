"""oprim.ext_pay_refund — refund a captured payment intent via a registered
obase.PaymentProvider (category="payment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import PaymentOprimError


async def ext_pay_refund(provider: str, *, intent_id: str, amount: int) -> dict[str, Any]:
    """Refund all or part of a captured payment intent.

    Args:
        provider: Provider name registered in ProviderRegistry (category="payment").
        intent_id: Intent ID of a previously captured payment.
        amount: Refund amount in minor currency units (e.g. cents).

    Returns:
        Provider's refund() response dict.

    Raises:
        PaymentOprimError: Provider not registered or the provider call failed
            (e.g. intent not captured, or amount exceeds the captured amount).
    """
    try:
        payment_provider = ProviderRegistry.get().generic("payment", provider)
    except ProviderNotFoundError as exc:
        raise PaymentOprimError(f"payment provider not found: {provider!r}", cause=exc) from exc

    try:
        return await payment_provider.refund(intent_id=intent_id, amount=amount)
    except Exception as exc:
        raise PaymentOprimError(f"refund failed: {exc}", cause=exc) from exc
