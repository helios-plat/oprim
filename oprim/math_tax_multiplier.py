"""oprim.math_tax_multiplier — convert a tax rate into a price multiplier
for tax-inclusive pricing calculations.
"""

from __future__ import annotations


def math_tax_multiplier(rate: float) -> float:
    """Convert a tax rate (0-100 percent scale, matching the
    `tax_rate.rate_percent` column convention used throughout this codebase)
    into a multiplier such that `price_with_tax = price * multiplier`.

    Args:
        rate: Tax rate on a 0-100 scale (e.g. 8.5 for 8.5%), matching
            `tax_rate.rate_percent`'s existing 0-100 CHECK constraint.

    Returns:
        `1 + rate / 100`.

    Example:
        >>> math_tax_multiplier(8.5)
        1.085
    """
    return 1 + rate / 100
