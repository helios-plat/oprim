"""Swing Failure Pattern (SFP) technical indicator.

Mathematical definition:
    Support_t = min(low_{t-lookback}, ..., low_{t-1})
    Resistance_t = max(high_{t-lookback}, ..., high_{t-1})
    ATR_t = wilder_atr(high, low, close, atr_period)

    Bullish SFP (+1.0):
        low_t < Support_t AND close_t >= Support_t
        AND (Support_t - low_t) >= min_pierce_atr * ATR_t
    Bearish SFP (-1.0):
        high_t > Resistance_t AND close_t <= Resistance_t
        AND (high_t - Resistance_t) >= min_pierce_atr * ATR_t
    Neutral (0.0): otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from oprim.technical._base import _to_array, _wilder_atr, _wrap


def swing_failure_pattern(
    highs: np.ndarray | pd.Series,
    lows: np.ndarray | pd.Series,
    closes: np.ndarray | pd.Series,
    *,
    lookback: int = 50,
    atr_period: int = 14,
    min_pierce_atr: float = 0.05,
) -> dict[str, np.ndarray | pd.Series]:
    """Detect Swing Failure Patterns (liquidity sweeps with reclaim).

    Parameters
    ----------
    highs : array-like
        High prices series.
    lows : array-like
        Low prices series.
    closes : array-like
        Close prices series.
    lookback : int
        Number of preceding bars to determine swing support and resistance.
    atr_period : int
        Wilder ATR period for pierce threshold normalization.
    min_pierce_atr : float
        Minimum pierce depth as a fraction of ATR (default 0.05).

    Returns
    -------
    dict with:
        'signal': array or Series in {-1.0, 0.0, 1.0}
        'support': trailing pivot support
        'resistance': trailing pivot resistance
    """
    h_arr, is_series, idx = _to_array(highs)
    l_arr, _, _ = _to_array(lows)
    c_arr, _, _ = _to_array(closes)

    n = len(c_arr)
    if n == 0:
        raise ValueError("prices must not be empty")
    if lookback <= 0 or atr_period <= 0:
        raise ValueError("lookback and atr_period must be positive integers")

    atr = _wilder_atr(h_arr, l_arr, c_arr, atr_period)

    signals = np.zeros(n, dtype=float)
    support = np.full(n, np.nan, dtype=float)
    resistance = np.full(n, np.nan, dtype=float)

    min_start = max(lookback, atr_period)
    for i in range(min_start, n):
        # Lookback window prior to current bar (strict causality)
        win_h = h_arr[i - lookback : i]
        win_l = l_arr[i - lookback : i]

        sup = float(np.min(win_l))
        res = float(np.max(win_h))
        support[i] = sup
        resistance[i] = res

        cur_atr = atr[i]
        if np.isnan(cur_atr) or cur_atr <= 0:
            cur_atr = max(h_arr[i] - l_arr[i], 0.01)

        threshold = min_pierce_atr * cur_atr
        bar_h = h_arr[i]
        bar_l = l_arr[i]
        bar_c = c_arr[i]

        # Bullish SFP: pierce below support and close back at or above support
        if bar_l < sup and bar_c >= sup and (sup - bar_l) >= threshold:
            signals[i] = 1.0
        # Bearish SFP: pierce above resistance and close back at or below resistance
        elif bar_h > res and bar_c <= res and (bar_h - res) >= threshold:
            signals[i] = -1.0

    return {
        "signal": _wrap(signals, is_series, idx),
        "support": _wrap(support, is_series, idx),
        "resistance": _wrap(resistance, is_series, idx),
    }
