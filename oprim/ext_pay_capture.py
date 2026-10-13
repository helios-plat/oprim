"""oprim.ext_pay_capture — capture a previously authorized payment intent via
a registered obase.PaymentProvider (category="payment").
"""

from __future__ import annotations

from typing import Any

from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import PaymentOprimError


async def ext_pay_capture(provider: str, *, intent_id: str) -> dict[str, Any]:
    """Capture an authorized payment intent.

    Args:
        provider: Provider name registered in ProviderRegistry (category="payment").
        intent_id: Intent ID returned by a prior `ext_pay_authorize` call.

    Returns:
        Provider's capture() response dict.

    Raises:
        PaymentOprimError: Provider not registered or the provider call failed
            (e.g. intent not in an authorized state).
    """
    try:
        payment_provider = ProviderRegistry.get().generic("payment", provider)
    except ProviderNotFoundError as exc:
        raise PaymentOprimError(f"payment provider not found: {provider!r}", cause=exc) from exc

    try:
        return await payment_provider.capture(intent_id=intent_id)
    except Exception as exc:
        raise PaymentOprimError(f"capture failed: {exc}", cause=exc) from exc
