"""Tests for oprim.risk.fee_edge_filter."""

import pytest

from oprim.risk.fee_edge_filter import fee_edge_filter


def test_high_atr_passes():
    r = fee_edge_filter(1000, atr_pct=0.02, taker_fee_rate=0.0005, edge_multiple=1.5)
    assert r["passes"] is True
    assert r["expected_gross"] == pytest.approx(20.0)
    assert r["fee"] == pytest.approx(1.0)


def test_tiny_atr_fails():
    # expected gross 0.05, fee round-trip 1.0, min 1.5 -> fails
    r = fee_edge_filter(1000, atr_pct=0.00005, taker_fee_rate=0.0005, edge_multiple=1.5)
    assert r["passes"] is False


def test_boundary_exact_pass():
    # expected_gross == min_gross -> passes (>=)
    # atr*N = fee*2*edge -> atr = 2*rate*edge
    rate, edge = 0.0005, 1.5
    atr = 2 * rate * edge
    r = fee_edge_filter(1000, atr_pct=atr, taker_fee_rate=rate, edge_multiple=edge)
    assert r["passes"] is True


def test_bad_params_raise():
    with pytest.raises(ValueError):
        fee_edge_filter(1000, atr_pct=0, taker_fee_rate=0.0005, edge_multiple=1.5)
    with pytest.raises(ValueError):
        fee_edge_filter(1000, atr_pct=0.02, taker_fee_rate=0.0005, edge_multiple=0)
