"""Average Directional Index (Wilder) — trend-strength indicator."""

from __future__ import annotations

import numpy as np


def adx(
    highs: np.ndarray | list,
    lows: np.ndarray | list,
    closes: np.ndarray | list,
    *,
    period: int = 14,
) -> float:
    """Average Directional Index (ADX) — Wilder's trend-strength measure.

    Computes +DI / -DI from directional movement, the DX = 100·|+DI − −DI| /
    (+DI + −DI), then Wilder-smooths DX into ADX. Returns the latest ADX value
    (0..100); higher = stronger trend (either direction), lower = ranging.

    Parameters
    ----------
    highs, lows, closes : array-like
        OHLC arrays of equal length, at least ``2*period + 1`` bars.
    period : int
        Smoothing period (default 14).

    Returns
    -------
    float
        Latest ADX value.

    Raises
    ------
    ValueError
        If fewer than ``2*period + 1`` bars are supplied.

    References
    ----------
    .. [1] Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
    """
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if n < 2 * period + 1:
        raise ValueError(f"need >= {2 * period + 1} bars, got {n}")

    up_move = h[1:] - h[:-1]
    down_move = lo[:-1] - lo[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum.reduce([h[1:] - lo[1:], np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])])

    def _wilder(x: np.ndarray) -> np.ndarray:
        out = np.empty(len(x))
        out[0] = x[:period].sum()
        for i in range(1, len(x)):
            out[i] = out[i - 1] - out[i - 1] / period + x[i]
        return out

    atr_s = _wilder(tr)
    plus_di = 100.0 * _wilder(plus_dm) / np.where(atr_s == 0, np.nan, atr_s)
    minus_di = 100.0 * _wilder(minus_dm) / np.where(atr_s == 0, np.nan, atr_s)
    denom = plus_di + minus_di
    dx = 100.0 * np.abs(plus_di - minus_di) / np.where(denom == 0, np.nan, denom)
    dx = np.nan_to_num(dx)

    # ADX = Wilder-smoothed DX over `period`
    adx_val = dx[:period].mean()
    for i in range(period, len(dx)):
        adx_val = (adx_val * (period - 1) + dx[i]) / period
    return float(adx_val)
