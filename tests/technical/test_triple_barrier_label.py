"""Tests for oprim.technical.triple_barrier_label (López de Prado labeling)."""

from __future__ import annotations

import numpy as np
import pytest

from oprim.technical.triple_barrier_label import triple_barrier_label


def test_upper_barrier_hit_gives_plus_one():
    closes = [100.0, 102.0, 104.0, 106.0]
    labels = triple_barrier_label(closes, take_profit_pct=0.03, stop_loss_pct=0.05, horizon=2)
    assert labels[0] == 1


def test_lower_barrier_hit_gives_minus_one():
    closes = [100.0, 96.0, 92.0, 90.0]
    labels = triple_barrier_label(closes, take_profit_pct=0.03, stop_loss_pct=0.03, horizon=2)
    assert labels[0] == -1


def test_vertical_barrier_gives_zero():
    closes = [100.0, 100.1, 100.0, 100.1, 100.0]
    labels = triple_barrier_label(closes, take_profit_pct=0.10, stop_loss_pct=0.10, horizon=2)
    assert labels[0] == 0


def test_earlier_barrier_wins():
    """Take-profit is crossed on bar 1, stop-loss on bar 2 -> label must be +1."""
    closes = [100.0, 103.0, 90.0]
    labels = triple_barrier_label(closes, take_profit_pct=0.02, stop_loss_pct=0.02, horizon=2)
    assert labels[0] == 1


def test_tail_bars_are_unlabelable():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    labels = triple_barrier_label(closes, take_profit_pct=0.5, stop_loss_pct=0.5, horizon=2)
    assert labels[-1] == 0
    assert labels[-2] == 0


def test_output_shape_and_dtype():
    closes = np.linspace(100.0, 120.0, 50)
    labels = triple_barrier_label(closes, take_profit_pct=0.02, stop_loss_pct=0.02, horizon=5)
    assert labels.shape == closes.shape
    assert labels.dtype == np.dtype(int)
    assert set(np.unique(labels)).issubset({-1, 0, 1})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"take_profit_pct": 0.0, "stop_loss_pct": 0.01, "horizon": 3}, "must be > 0"),
        ({"take_profit_pct": 0.01, "stop_loss_pct": -0.01, "horizon": 3}, "must be > 0"),
        ({"take_profit_pct": 0.01, "stop_loss_pct": 0.01, "horizon": 0}, "horizon must be >= 1"),
    ],
)
def test_rejects_bad_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        triple_barrier_label([100.0, 101.0, 102.0], **kwargs)


def test_accepts_list_and_array_equivalently():
    closes = [100.0, 101.5, 99.0, 103.0, 101.0, 98.0]
    as_list = triple_barrier_label(closes, take_profit_pct=0.02, stop_loss_pct=0.02, horizon=2)
    as_array = triple_barrier_label(
        np.asarray(closes), take_profit_pct=0.02, stop_loss_pct=0.02, horizon=2
    )
    assert np.array_equal(as_list, as_array)
