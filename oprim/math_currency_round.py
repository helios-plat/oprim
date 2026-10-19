"""oprim.math_currency_round — round a decimal-currency amount into minor
units (cents), avoiding float-imprecision artifacts from naive `round()`.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def math_currency_round(amount: float, *, decimals: int = 2) -> int:
    """Round `amount` to `decimals` places and return it as an integer of
    minor currency units (e.g. cents when decimals=2).

    Args:
        amount: Decimal currency amount (e.g. dollars).
        decimals: Number of fractional digits the currency uses.

    Returns:
        Integer minor-unit amount (`round(amount, decimals) * 10**decimals`),
        using ROUND_HALF_UP semantics standard for currency.

    Example:
        >>> math_currency_round(19.995)
        2000
        >>> math_currency_round(19.994)
        1999
    """
    quantum = Decimal(1).scaleb(-decimals)
    minor_unit_scale = Decimal(10) ** decimals
    rounded = Decimal(str(amount)).quantize(quantum, rounding=ROUND_HALF_UP)
    return int(rounded * minor_unit_scale)
