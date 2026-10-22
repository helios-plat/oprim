"""Unit tests for oprim.technical.sfp (Swing Failure Pattern)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oprim.technical.sfp import swing_failure_pattern


def test_sfp_empty_and_invalid_args():
    with pytest.raises(ValueError, match="prices must not be empty"):
        swing_failure_pattern(np.array([]), np.array([]), np.array([]))

    with pytest.raises(ValueError, match="positive integers"):
        swing_failure_pattern(np.array([1.0]), np.array([1.0]), np.array([1.0]), lookback=0)


def test_sfp_bullish_sweep_and_reclaim():
    """Verify bullish SFP: price wicks below support then closes above support."""
    n = 60
    # Flat base between 100 and 110 for first 55 bars
    highs = np.full(n, 110.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 105.0)

    # Bar 56: sweep low down to 98.0 (support is 100.0), close back at 101.0
    lows[56] = 98.0
    closes[56] = 101.0

    res = swing_failure_pattern(highs, lows, closes, lookback=50, atr_period=14)
    signals = res["signal"]
    assert signals[56] == 1.0  # Bullish SFP detected
    assert res["support"][56] == pytest.approx(100.0)


def test_sfp_bearish_sweep_and_reclaim():
    """Verify bearish SFP: price wicks above resistance then closes below resistance."""
    n = 60
    highs = np.full(n, 110.0)
    lows = np.full(n, 100.0)
    closes = np.full(n, 105.0)

    # Bar 56: sweep high up to 112.0 (resistance is 110.0), close back at 109.0
    highs[56] = 112.0
    closes[56] = 109.0

    res = swing_failure_pattern(highs, lows, closes, lookback=50, atr_period=14)
    signals = res["signal"]
    assert signals[56] == -1.0  # Bearish SFP detected
    assert res["resistance"][56] == pytest.approx(110.0)


def test_sfp_pandas_series_preservation():
    idx = pd.date_range("2024-01-01", periods=60, freq="h")
    high_s = pd.Series(np.full(60, 110.0), index=idx)
    low_s = pd.Series(np.full(60, 100.0), index=idx)
    close_s = pd.Series(np.full(60, 105.0), index=idx)

    res = swing_failure_pattern(high_s, low_s, close_s, lookback=50)
    assert isinstance(res["signal"], pd.Series)
    assert list(res["signal"].index) == list(idx)
