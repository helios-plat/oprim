"""Tests for oprim.technical.adx."""

import numpy as np
import pytest

from oprim.technical.adx import adx


def test_strong_trend_high_adx():
    h = np.cumsum(np.abs(np.random.default_rng(1).normal(0.5, 0.2, 100))) + 100
    lo = h - 0.5
    c = h - 0.2
    assert adx(h, lo, c, period=14) > 40


def test_choppy_low_adx():
    rng = np.random.default_rng(2)
    c = 100 + rng.normal(0, 1, 100)
    h = c + 0.5
    lo = c - 0.5
    assert adx(h, lo, c, period=14) < 30


def test_range_bounded_0_100():
    rng = np.random.default_rng(3)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 100))
    v = adx(c + 0.3, c - 0.3, c, period=14)
    assert 0.0 <= v <= 100.0


def test_insufficient_bars_raises():
    with pytest.raises(ValueError):
        adx([100] * 10, [99] * 10, [99.5] * 10, period=14)
