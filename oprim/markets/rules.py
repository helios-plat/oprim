"""Market regulatory rule primitives (parameterized).

Tax / settlement / commission rules that vary by market. All parameters
are explicit; oprim does not hardcode any market's rates.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

STABILITY = "experimental"


def stamp_tax(
    amount: float,
    rate: float,
    direction: Literal["buy", "sell", "both"],
) -> float:
    """Compute stamp tax on a trade.

    Parameters
    ----------
    amount : transaction amount in market currency
    rate : stamp tax rate (e.g. 0.0005 for A-share post-2023-08-28)
    direction : "buy" / "sell" / "both" - which side(s) the tax applies to.
                "both" means tax is levied on each side; caller provides
                per-side amount and this function returns per-side tax.

    Returns
    -------
    Tax amount (always >= 0)

    Raises
    ------
    ValueError : if rate < 0

    Notes
    -----
    Examples of real-world rates (caller provides):
    - A-share sell only: 0.0005 (post-2023-08-28), 0.001 (pre)
    - HK both sides: 0.001 each
    - UK SDRT buy only: 0.005

    Reference
    ---------
    Market microstructure transaction cost literature.
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    if direction not in ("buy", "sell", "both"):
        raise ValueError(f"direction must be 'buy', 'sell', or 'both', got {direction!r}")
    if amount <= 0:
        return 0.0
    return amount * rate


def _count_business_days(from_date: date, to_date: date) -> int:
    """Count weekday business days strictly after from_date up to and including to_date."""
    if to_date <= from_date:
        return 0
    count = 0
    d = from_date + timedelta(days=1)
    while d <= to_date:
        if d.weekday() < 5:
            count += 1
        d += timedelta(days=1)
    return count


def t_plus_n_blocked(
    buy_date: date,
    sell_date: date,
    n: int,
) -> bool:
    """Check if T+N rule blocks the sell.

    Parameters
    ----------
    buy_date : date of purchase
    sell_date : proposed sell date
    n : T+N settlement period (0 = T+0 = same day allowed,
        1 = T+1 = next business day required, etc.)

    Returns
    -------
    True if sell is blocked (business days between buy and sell < n)

    Raises
    ------
    ValueError : if n < 0

    Examples
    --------
    >>> t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 19), 1)  # A-share T+1
    True
    >>> t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 20), 1)
    False
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    business_days_elapsed = _count_business_days(buy_date, sell_date)
    return business_days_elapsed < n


def commission(
    amount: float,
    rate: float,
    min_fee: float = 0.0,
) -> float:
    """Compute commission with optional minimum fee floor.

    Returns max(amount * rate, min_fee) if amount > 0, else 0.

    Parameters
    ----------
    amount : transaction amount
    rate : commission rate (e.g. 0.00025 for A-share retail)
    min_fee : minimum fee floor (e.g. 5.0 for A-share)

    Returns
    -------
    Commission amount >= 0

    Raises
    ------
    ValueError : if rate < 0 or min_fee < 0
    """
    if rate < 0:
        raise ValueError(f"rate must be non-negative, got {rate}")
    if min_fee < 0:
        raise ValueError(f"min_fee must be non-negative, got {min_fee}")
    if amount <= 0:
        return 0.0
    return max(amount * rate, min_fee)
