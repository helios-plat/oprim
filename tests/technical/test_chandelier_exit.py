"""Tests for oprim.chandelier_exit."""

import numpy as np
import pandas as pd
import pytest

from oprim.technical.exits import chandelier_exit


def _make_ohlc(n=50, seed=42):
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    return highs, lows, closes


def test_chandelier_basic():
    h, l, c = _make_ohlc()
    result = chandelier_exit(h, l, c, period=10, multiplier=3.0)
    assert set(result.keys()) == {"long_exit", "short_exit"}
    assert len(result["long_exit"]) == 50


def test_chandelier_long_exit_below_recent_high():
    h, l, c = _make_ohlc(50)
    result = chandelier_exit(h, l, c, period=10, multiplier=3.0)
    long_exit = result["long_exit"]
    # long_exit should be below rolling high by at least multiplier * small_atr
    valid_idx = np.where(np.isfinite(long_exit))[0]
    for t in valid_idx:
        roll_high = np.max(h[max(0, t - 9) : t + 1])
        assert long_exit[t] <= roll_high


def test_chandelier_short_exit_above_recent_low():
    h, l, c = _make_ohlc(50)
    result = chandelier_exit(h, l, c, period=10, multiplier=3.0)
    short_exit = result["short_exit"]
    valid_idx = np.where(np.isfinite(short_exit))[0]
    for t in valid_idx:
        roll_low = np.min(l[max(0, t - 9) : t + 1])
        assert short_exit[t] >= roll_low


def test_chandelier_mismatched_length_raises():
    h, l, c = _make_ohlc(50)
    with pytest.raises(ValueError):
        chandelier_exit(h[:40], l, c)
    with pytest.raises(ValueError):
        chandelier_exit(h, l[:40], c)


def test_chandelier_invalid_period():
    h, l, c = _make_ohlc()
    with pytest.raises(ValueError):
        chandelier_exit(h, l, c, period=0)
    with pytest.raises(ValueError):
        chandelier_exit(h, l, c, period=-5)


def test_chandelier_invalid_multiplier():
    h, l, c = _make_ohlc()
    with pytest.raises(ValueError):
        chandelier_exit(h, l, c, multiplier=0.0)
    with pytest.raises(ValueError):
        chandelier_exit(h, l, c, multiplier=-1.0)


def test_chandelier_preserves_series():
    h, l, c = _make_ohlc(50)
    idx = pd.date_range("2024-01-01", periods=50)
    result = chandelier_exit(
        pd.Series(h, index=idx),
        pd.Series(l, index=idx),
        pd.Series(c, index=idx),
    )
    assert isinstance(result["long_exit"], pd.Series)
    assert list(result["long_exit"].index) == list(idx)


@pytest.mark.academic_reference
def test_chandelier_matches_published_example():
    """Chandelier Exit matches manual computation of highest_high - mult*ATR.

    Reference: Le Beau (1990s) definition. Verified against manual ATR + rolling max.
    """
    rng = np.random.default_rng(77)
    n = 100
    closes = 100.0 + np.cumsum(rng.normal(0, 1, n))
    highs = closes + rng.uniform(0.5, 2.0, n)
    lows = closes - rng.uniform(0.5, 2.0, n)
    period = 22
    multiplier = 3.0

    result = chandelier_exit(highs, lows, closes, period=period, multiplier=multiplier)

    # Manual: ATR via Wilder, rolling max/min
    from oprim.technical._base import _wilder_atr
    atr_arr = _wilder_atr(highs, lows, closes, period)
    roll_high = pd.Series(highs).rolling(period).max().to_numpy()
    roll_low = pd.Series(lows).rolling(period).min().to_numpy()
    expected_long = roll_high - multiplier * atr_arr
    expected_short = roll_low + multiplier * atr_arr

    valid = np.isfinite(expected_long) & np.isfinite(np.asarray(result["long_exit"]))
    np.testing.assert_allclose(
        np.asarray(result["long_exit"])[valid],
        expected_long[valid],
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(result["short_exit"])[valid],
        expected_short[valid],
        rtol=1e-8,
    )


def test_chandelier_too_short_series():
    """_wilder_atr returns all NaN when n-1 < period."""
    h = np.array([1.1, 1.2])
    l = np.array([0.9, 1.0])
    c = np.array([1.0, 1.1])
    result = chandelier_exit(h, l, c, period=5)
    # Too short for ATR: all NaN
    assert all(np.isnan(result["long_exit"]))


def test_chandelier_single_bar_too_short():
    """_wilder_atr n < 2 returns all NaN."""
    h = np.array([1.1])
    l = np.array([0.9])
    c = np.array([1.0])
    result = chandelier_exit(h, l, c, period=1)
    assert all(np.isnan(result["long_exit"]))
