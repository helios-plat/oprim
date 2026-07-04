"""Tests for oprim.journal_discipline_score (ported from Tide's
TestComputeDisciplineBreakdown, 11 tests; DB-bound score_from_entry_id/route
tests stay in Tide). Breakdown key renamed holding_days_within_50pct_sell ->
holding_days_within_tolerance_sell to match the generalized holding_tolerance
parameter (default 0.5 == the original hardcoded ±50%).
"""

from __future__ import annotations

from datetime import date

import pytest

from oprim.journal_discipline_score import journal_discipline_score


def test_all_rules_buy() -> None:
    entry = {
        "side": "buy",
        "entry_reason": "趋势共振",
        "plan_match": True,
        "confidence_level": 5,
    }
    result = journal_discipline_score(entry)
    assert result["total_score"] == pytest.approx(3.0)
    bd = result["breakdown"]
    assert bd["entry_reason_present"] == 1.0
    assert bd["plan_match"] == 1.0
    assert bd["confidence_level_ge3"] == 1.0
    assert bd["exit_reason_present_sell"] == 0.0
    assert bd["holding_days_within_tolerance_sell"] == 0.0


def test_all_rules_sell_full_score() -> None:
    entry = {
        "side": "sell",
        "entry_reason": "良好信号",
        "plan_match": True,
        "confidence_level": 4,
        "exit_reason": "止盈",
        "expected_holding_days": 20,
        "trade_date": date(2026, 5, 21),
        "entry_date": date(2026, 5, 1),
    }
    result = journal_discipline_score(entry)
    assert result["total_score"] == pytest.approx(5.0)


def test_no_rules_sell_zero() -> None:
    entry = {"side": "sell"}
    result = journal_discipline_score(entry)
    assert result["total_score"] == pytest.approx(0.0)
    assert all(v == 0.0 for v in result["breakdown"].values())


def test_breakdown_keys_present() -> None:
    result = journal_discipline_score({})
    bd = result["breakdown"]
    assert "entry_reason_present" in bd
    assert "plan_match" in bd
    assert "confidence_level_ge3" in bd
    assert "exit_reason_present_sell" in bd
    assert "holding_days_within_tolerance_sell" in bd


def test_partial_score_buy() -> None:
    entry = {
        "side": "buy",
        "entry_reason": "MA突破",
        "plan_match": False,
        "confidence_level": 2,
    }
    result = journal_discipline_score(entry)
    assert result["total_score"] == pytest.approx(1.0)  # only entry_reason


def test_confidence_exactly_3() -> None:
    entry = {"side": "buy", "confidence_level": 3}
    result = journal_discipline_score(entry)
    assert result["breakdown"]["confidence_level_ge3"] == 1.0


def test_holding_days_boundary_50pct_above() -> None:
    """Actual days ratio == 0.5 (boundary) → within default ±50% tolerance."""
    entry = {
        "side": "sell",
        "expected_holding_days": 10,
        "trade_date": date(2026, 5, 16),  # 15 days after
        "entry_date": date(2026, 5, 1),
    }
    result = journal_discipline_score(entry)
    assert result["breakdown"]["holding_days_within_tolerance_sell"] == 1.0


def test_holding_days_exactly_at_50pct_boundary() -> None:
    """actual = 1.5x expected -> ratio = 0.5 -> pass."""
    entry = {
        "side": "sell",
        "expected_holding_days": 10,
        "trade_date": date(2026, 5, 16),  # 15 days
        "entry_date": date(2026, 5, 1),
    }
    result = journal_discipline_score(entry)
    assert result["breakdown"]["holding_days_within_tolerance_sell"] == 1.0


def test_holding_days_exceeds_50pct() -> None:
    """actual = 2x expected -> ratio = 1.0 > 0.5 -> no score."""
    entry = {
        "side": "sell",
        "expected_holding_days": 10,
        "trade_date": date(2026, 5, 21),  # 20 days
        "entry_date": date(2026, 5, 1),
    }
    result = journal_discipline_score(entry)
    assert result["breakdown"]["holding_days_within_tolerance_sell"] == 0.0


def test_string_date_parsing() -> None:
    """trade_date and entry_date as strings should be parsed."""
    entry = {
        "side": "sell",
        "expected_holding_days": 30,
        "trade_date": "2026-05-31",
        "entry_date": "2026-05-01",
    }
    result = journal_discipline_score(entry)
    assert result["breakdown"]["holding_days_within_tolerance_sell"] == 1.0


def test_custom_holding_tolerance() -> None:
    """A tighter ±20% tolerance rejects what the default ±50% would accept."""
    entry = {
        "side": "sell",
        "expected_holding_days": 10,
        "trade_date": date(2026, 5, 16),  # 15 days -> ratio 0.5
        "entry_date": date(2026, 5, 1),
    }
    default_result = journal_discipline_score(entry)
    tight_result = journal_discipline_score(entry, holding_tolerance=0.2)
    assert default_result["breakdown"]["holding_days_within_tolerance_sell"] == 1.0
    assert tight_result["breakdown"]["holding_days_within_tolerance_sell"] == 0.0
