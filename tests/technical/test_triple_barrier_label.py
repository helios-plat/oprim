"""Tests for oprim.technical.triple_barrier_label."""

import numpy as np
import pytest

from oprim.technical.triple_barrier_label import triple_barrier_label


def test_rising_series_labels_long():
    c = list(100 * np.exp(np.cumsum(np.full(200, 0.001))))
    lab = triple_barrier_label(c, take_profit_pct=0.015, stop_loss_pct=0.010, horizon=30)
    assert (lab[:-30] == 1).all()
    assert (lab[-30:] == 0).all()  # unlabelable tail


def test_falling_series_labels_short():
    c = list(100 * np.exp(np.cumsum(np.full(200, -0.001))))
    lab = triple_barrier_label(c, take_profit_pct=0.015, stop_loss_pct=0.010, horizon=30)
    assert (lab[:-30] == -1).all()


def test_flat_series_labels_neutral():
    c = [100.0] * 100
    lab = triple_barrier_label(c, take_profit_pct=0.015, stop_loss_pct=0.010, horizon=20)
    assert (lab == 0).all()


def test_length_preserved():
    c = list(np.random.default_rng(0).normal(100, 1, 150))
    lab = triple_barrier_label(c, take_profit_pct=0.02, stop_loss_pct=0.02, horizon=10)
    assert len(lab) == 150


def test_bad_params_raise():
    c = [100.0] * 50
    with pytest.raises(ValueError):
        triple_barrier_label(c, take_profit_pct=0, stop_loss_pct=0.01, horizon=10)
    with pytest.raises(ValueError):
        triple_barrier_label(c, take_profit_pct=0.01, stop_loss_pct=0.01, horizon=0)
