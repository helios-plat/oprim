"""Daily price limit detection primitives.

Market-agnostic. Each market provides its own limit_pct:
- A-share: 0.10 (main board) / 0.20 (ChiNext/STAR) / 0.30 (BSE)
- HK: 0.10 / 0.20 (varies by stock type)
- Taiwan: 0.10
- Korea: 0.30

The function only checks if today's close is at the daily limit;
market-specific limit_pct lookup is Layer 4 responsibility.
"""

from __future__ import annotations

STABILITY = "experimental"


def detect_daily_limit_up(
    close: float,
    prev_close: float,
    limit_pct: float,
    tolerance: float = 0.0005,
) -> bool:
    """Detect if today's close hit the daily upper limit.

    Parameters
    ----------
    close : today's closing price
    prev_close : previous trading day's closing price
    limit_pct : daily limit fraction (e.g. 0.10 = 10%)
    tolerance : floating-point tolerance for comparison (default 0.0005)

    Returns
    -------
    True if close >= prev_close * (1 + limit_pct) - tolerance

    Raises
    ------
    ValueError : if prev_close <= 0 or limit_pct < 0

    Examples
    --------
    >>> detect_daily_limit_up(11.0, 10.0, 0.10)  # A-share main board
    True
    >>> detect_daily_limit_up(12.0, 10.0, 0.20)  # A-share ChiNext
    True
    """
    if prev_close <= 0:
        raise ValueError(f"prev_close must be positive, got {prev_close}")
    if limit_pct < 0:
        raise ValueError(f"limit_pct must be non-negative, got {limit_pct}")

    limit_price = prev_close * (1.0 + limit_pct)
    return close >= limit_price - tolerance


def detect_daily_limit_down(
    close: float,
    prev_close: float,
    limit_pct: float,
    tolerance: float = 0.0005,
) -> bool:
    """Detect if today's close hit the daily lower limit. Symmetric to limit_up.

    Parameters
    ----------
    close : today's closing price
    prev_close : previous trading day's closing price
    limit_pct : daily limit fraction (e.g. 0.10 = 10%)
    tolerance : floating-point tolerance for comparison (default 0.0005)

    Returns
    -------
    True if close <= prev_close * (1 - limit_pct) + tolerance

    Raises
    ------
    ValueError : if prev_close <= 0 or limit_pct < 0
    """
    if prev_close <= 0:
        raise ValueError(f"prev_close must be positive, got {prev_close}")
    if limit_pct < 0:
        raise ValueError(f"limit_pct must be non-negative, got {limit_pct}")

    limit_price = prev_close * (1.0 - limit_pct)
    return close <= limit_price + tolerance


def seal_strength(seal_amount: float, total_volume: float) -> float:
    """Compute the seal-order strength ratio.

    Returns seal_amount / total_volume, clamped to [0, +inf).

    Used in limit-up board strength analysis: higher seal-to-volume ratio
    indicates stronger demand pressure at the limit.

    Parameters
    ----------
    seal_amount : outstanding buy orders at the limit-up price
    total_volume : total traded volume for the day

    Returns
    -------
    Ratio >= 0. Returns 0 if total_volume <= 0 or seal_amount < 0.

    Reference: market microstructure literature on order book imbalance.
    """
    if total_volume <= 0 or seal_amount < 0:
        return 0.0
    return seal_amount / total_volume
