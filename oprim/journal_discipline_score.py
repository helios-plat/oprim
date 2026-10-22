"""交易日志纪律分 — 单条日志 0-5 分, 5 条布尔规则.

Ported from Tide's ``domain.journal.scoring`` (``compute_discipline_score`` +
``compute_discipline_breakdown``), merged into one function returning the
breakdown (a superset of the plain score). Generalized: the original hardcoded
±50% holding-period tolerance (rule 5) is now a ``holding_tolerance`` parameter.
"""

from __future__ import annotations

from datetime import date
from typing import Any


def journal_discipline_score(
    entry: dict[str, Any],
    *,
    holding_tolerance: float = 0.5,
) -> dict[str, Any]:
    """Compute a journal entry's discipline score (0-5) + per-rule breakdown.

    Rules (each contributes +1):
        1. entry_reason is present (not None/empty)
        2. plan_match is True
        3. confidence_level >= 3
        4. side == 'sell' and exit_reason is present
        5. side == 'sell' and expected_holding_days set and actual days within
           ±holding_tolerance of expected (e.g. 0.5 == ±50%)

    Args:
        entry: Journal entry dict (entry_reason, plan_match, confidence_level,
            side, exit_reason, expected_holding_days, trade_date, entry_date).
        holding_tolerance: Fractional tolerance for rule 5 (default ±50%).

    Returns:
        dict with total_score (0-5) and breakdown (per-rule 0/1 contributions).
    """
    rules: dict[str, float] = {}

    entry_reason = entry.get("entry_reason")
    rules["entry_reason_present"] = (
        1.0 if (entry_reason is not None and str(entry_reason).strip() != "") else 0.0
    )

    rules["plan_match"] = 1.0 if entry.get("plan_match") is True else 0.0

    confidence = entry.get("confidence_level")
    rules["confidence_level_ge3"] = (
        1.0 if (confidence is not None and int(confidence) >= 3) else 0.0
    )

    side = str(entry.get("side", "")).lower()
    exit_reason = entry.get("exit_reason")
    rules["exit_reason_present_sell"] = 0.0
    rules["holding_days_within_tolerance_sell"] = 0.0

    if side == "sell":
        rules["exit_reason_present_sell"] = (
            1.0 if (exit_reason is not None and str(exit_reason).strip() != "") else 0.0
        )
        expected_days = entry.get("expected_holding_days")
        if expected_days is not None:
            trade_date = entry.get("trade_date")
            entry_date_val = entry.get("entry_date")
            actual_days: int | None = None
            if trade_date is not None and entry_date_val is not None:
                td = trade_date if isinstance(trade_date, date) else _parse_date(trade_date)
                ed = (
                    entry_date_val
                    if isinstance(entry_date_val, date)
                    else _parse_date(entry_date_val)
                )
                if td is not None and ed is not None:
                    actual_days = (td - ed).days
            if actual_days is not None and actual_days >= 0:
                exp = int(expected_days)
                within_tolerance = exp > 0 and abs(actual_days - exp) / exp <= holding_tolerance
                exact_zero_match = exp == 0 and actual_days == 0
                if within_tolerance or exact_zero_match:
                    rules["holding_days_within_tolerance_sell"] = 1.0

    total = sum(rules.values())
    return {"total_score": total, "breakdown": rules}


def _parse_date(val: Any) -> date | None:
    """Parse a date value, returning None if unparseable."""
    if isinstance(val, date):
        return val
    try:
        from datetime import datetime as dt

        return dt.strptime(str(val), "%Y-%m-%d").date()
    except Exception:
        return None
