"""Tests for oprim.risk.atr_position_cap."""

import pytest

from oprim.risk.atr_position_cap import atr_position_cap


def test_basic_clamp_within_bounds():
    result = atr_position_cap(atr_pct=0.02, risk_budget=0.01, min_position=0.005, max_position=1.0)
    assert result == pytest.approx(0.01 / 0.02)


def test_high_volatility_hits_min_floor():
    result = atr_position_cap(atr_pct=10.0, risk_budget=0.01, min_position=0.005, max_position=0.20)
    assert result == 0.005


def test_low_volatility_hits_max_ceiling():
    result = atr_position_cap(
        atr_pct=0.0001, risk_budget=0.01, min_position=0.005, max_position=0.20
    )
    assert result == 0.20


def test_inverse_relationship_doubling_atr_halves_cap():
    low = atr_position_cap(atr_pct=0.01, risk_budget=0.01, min_position=0.0, max_position=10.0)
    high = atr_position_cap(atr_pct=0.02, risk_budget=0.01, min_position=0.0, max_position=10.0)
    assert high == pytest.approx(low / 2)


def test_atr_pct_zero_raises():
    with pytest.raises(ValueError):
        atr_position_cap(atr_pct=0.0, risk_budget=0.01, min_position=0.0, max_position=1.0)


def test_atr_pct_negative_raises():
    with pytest.raises(ValueError):
        atr_position_cap(atr_pct=-0.01, risk_budget=0.01, min_position=0.0, max_position=1.0)


def test_risk_budget_zero_raises():
    with pytest.raises(ValueError):
        atr_position_cap(atr_pct=0.01, risk_budget=0.0, min_position=0.0, max_position=1.0)


def test_min_greater_than_max_raises():
    with pytest.raises(ValueError):
        atr_position_cap(atr_pct=0.01, risk_budget=0.01, min_position=0.5, max_position=0.1)
