"""Business day calendar primitives.

Pure date arithmetic. Skips weekends only. National holidays are
Layer 4 concern (each market has its own holiday list).

Reference: pandas.tseries.offsets.BusinessDay (for comparison only;
3O doesn't depend on pandas tseries).
"""

from __future__ import annotations

from datetime import date, timedelta

STABILITY = "experimental"


def is_business_day(d: date) -> bool:
    """Return True if d is Monday-Friday.

    Parameters
    ----------
    d : date to check
    """
    return d.weekday() < 5


def prev_business_day(d: date, n: int = 1) -> date:
    """Return the n-th previous business day (Mon-Fri).

    Parameters
    ----------
    d : reference date
    n : number of business days to subtract (n >= 1)

    Returns
    -------
    date n business days before d

    Raises
    ------
    ValueError if n < 1

    Examples
    --------
    >>> prev_business_day(date(2026, 5, 19), 1)  # Tuesday -> Monday
    datetime.date(2026, 5, 18)
    >>> prev_business_day(date(2026, 5, 19), 3)  # Tuesday -> previous Thursday
    datetime.date(2026, 5, 14)
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    current = d
    count = 0
    while count < n:
        current -= timedelta(days=1)
        if current.weekday() < 5:
            count += 1

    return current
