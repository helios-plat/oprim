"""Average Directional Index (Wilder) — scalar latest-value view of ``adx_series``.

``adx_series`` in :mod:`oprim.technical.trend` is the canonical ADX
implementation (full +DI / -DI / ADX series). Several consumers only need the
latest value and would otherwise re-derive the Wilder smoothing themselves,
which is exactly the duplicate-implementation drift SPEC §1.4 forbids.

So this module is a **delegating view**, not a second implementation: it calls
``adx_series`` and returns the last valid ADX reading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oprim.technical.trend import adx_series

__all__ = ["adx"]


def adx(
    highs: np.ndarray | pd.Series | list,
    lows: np.ndarray | pd.Series | list,
    closes: np.ndarray | pd.Series | list,
    *,
    period: int = 14,
) -> float:
    """Latest Average Directional Index (ADX) value.

    ADX measures trend *strength* regardless of direction: ``> 25`` trending,
    ``< 20`` choppy/ranging. This is the scalar convenience form of
    :func:`oprim.technical.trend.adx_series` — it delegates to that canonical
    implementation and returns the final valid reading.

    Parameters
    ----------
    highs, lows, closes : array-like
        OHLC series of equal length. At least ``2 * period + 1`` bars are
        needed before the first valid ADX value exists.
    period : int
        Wilder smoothing period (default 14).

    Returns
    -------
    float
        Latest ADX value in ``0..100``. ``nan`` when the input is too short to
        produce a valid reading.

    Raises
    ------
    ValueError
        If the arrays have different lengths, are empty, or ``period <= 0``.

    References
    ----------
    .. [1] Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
    """
    series = adx_series(highs, lows, closes, period=period)
    values = np.asarray(series["adx"], dtype=float)
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan")
    return float(valid[-1])
