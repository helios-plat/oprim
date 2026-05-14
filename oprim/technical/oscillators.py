"""Oscillator technical indicators (RSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oprim.technical._base import _to_array, _wrap


def rsi_normalized(
    prices: np.ndarray | pd.Series,
    *,
    period: int = 14,
) -> np.ndarray | pd.Series:
    """Relative Strength Index, normalized to [0, 1].

    Mathematical definition (Wilder smoothing):
        delta_t = P_t - P_{t-1}
        gain_t = max(delta_t, 0)
        loss_t = max(-delta_t, 0)

        avg_gain_0 = mean(gain[0:period])
        avg_gain_t = (avg_gain_{t-1} * (period - 1) + gain_t) / period

        rs_t = avg_gain_t / avg_loss_t
        rsi_t = 1 - 1 / (1 + rs_t)  in [0, 1]

    Standard RSI = rsi_normalized * 100.
    When avg_loss = 0: rsi = 1.0 (pure uptrend, no losses).
    First `period` positions are NaN.
    Input NaN propagates to output.

    Reference: Wilder (1978), "New Concepts in Technical Trading Systems".

    Parameters
    ----------
    prices : array-like
        Time-ordered price series (at least period+1 bars recommended).
    period : int
        Wilder smoothing period (default 14).

    Returns
    -------
    Same type as input, values in [0, 1], NaN for first period positions.

    Raises
    ------
    ValueError
        If prices is empty or period <= 0.
    """
    arr, is_series, idx = _to_array(prices)
    n = len(arr)
    if n == 0:
        raise ValueError("prices must not be empty")
    if not isinstance(period, int) or period <= 0:
        raise ValueError(f"period must be a positive integer, got {period!r}")

    out = np.full(n, np.nan)
    if n < period + 1:
        return _wrap(out, is_series, idx)

    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Seed: SMA of first `period` differences → RSI at index `period`
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    if avg_loss == 0.0:
        out[period] = 1.0
    else:
        out[period] = 1.0 - 1.0 / (1.0 + avg_gain / avg_loss)

    # Wilder smoothing for remaining positions
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            out[i + 1] = 1.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 1.0 - 1.0 / (1.0 + rs)

    return _wrap(out, is_series, idx)
