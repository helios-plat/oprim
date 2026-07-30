"""oprim.math_calc_percentage — apply a percentage rate to an integer base
amount (e.g. minor-unit cents), returning an un-rounded float.
"""

from __future__ import annotations


def math_calc_percentage(base: int, *, percent: float) -> float:
    """Compute `percent`% of `base`.

    Callers that need an integer minor-unit result should round the output
    themselves (e.g. via `math_currency_round`) — this atom stays a pure,
    un-rounded calculation so it composes with any rounding policy.

    Args:
        base: Base amount (e.g. cart subtotal in cents).
        percent: Percentage rate on a 0-100 scale (e.g. 8.5 for 8.5%).

    Returns:
        `base * percent / 100`.

    Example:
        >>> math_calc_percentage(1000, percent=8.5)
        85.0
    """
    return base * percent / 100
