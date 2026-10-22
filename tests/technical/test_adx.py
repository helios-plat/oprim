"""Single-source guard for the two ADX surfaces.

``oprim.technical.trend.adx_series`` is the canonical implementation;
``oprim.technical.adx`` is a delegating scalar view of it (SPEC §1.4 — one
implementation, the rest re-export/delegate). These tests fail if the two paths
ever drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oprim.technical.adx import adx
from oprim.technical.trend import adx_series


def _ohlc(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 0.5, n))
    highs = closes + rng.random(n)
    lows = closes - rng.random(n)
    return highs, lows, closes


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_adx_matches_last_valid_adx_series_value(seed):
    highs, lows, closes = _ohlc(seed=seed)
    series = np.asarray(adx_series(highs, lows, closes, period=14)["adx"], dtype=float)
    expected = series[~np.isnan(series)][-1]
    assert adx(highs, lows, closes, period=14) == pytest.approx(expected)


def test_adx_agrees_on_pandas_input():
    highs, lows, closes = _ohlc()
    hs = pd.Series(highs)
    ls = pd.Series(lows)
    cs = pd.Series(closes)
    assert adx(hs, ls, cs) == pytest.approx(adx(highs, lows, closes))


def test_adx_range_is_zero_to_hundred():
    highs, lows, closes = _ohlc(n=300, seed=9)
    value = adx(highs, lows, closes)
    assert 0.0 <= value <= 100.0


def test_adx_too_short_returns_nan():
    highs, lows, closes = _ohlc(n=10)
    assert np.isnan(adx(highs, lows, closes, period=14))


def test_adx_rejects_mismatched_lengths():
    highs, lows, closes = _ohlc(n=100)
    with pytest.raises(ValueError, match="same length"):
        adx(highs, lows[:-1], closes)


def test_adx_rejects_bad_period():
    highs, lows, closes = _ohlc(n=100)
    with pytest.raises(ValueError, match="period must be a positive integer"):
        adx(highs, lows, closes, period=0)


def test_adx_has_no_sibling_oprim_call():
    """adx may only depend on the shared technical base + its canonical source."""
    import ast
    import importlib
    from pathlib import Path

    mod = importlib.import_module("oprim.technical.adx")
    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    allowed = {"oprim.technical.trend"}
    intra_oprim = {m for m in imported if m.startswith("oprim.")}
    assert intra_oprim <= allowed, f"unexpected oprim imports: {intra_oprim - allowed}"
