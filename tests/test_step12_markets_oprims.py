"""Tests for Step-12 markets-related oprims (oprim 2.22.0)."""

from __future__ import annotations

from datetime import date

import pytest

from oprim.compute_commission import compute_commission
from oprim.compute_stamp_tax import compute_stamp_tax
from oprim.detect_daily_limit_down import detect_daily_limit_down
from oprim.detect_daily_limit_up import detect_daily_limit_up
from oprim.t_plus_n_blocked import t_plus_n_blocked


# ---------------------------------------------------------------------------
# detect_daily_limit_up — ≥7 tests
# ---------------------------------------------------------------------------


class TestDetectDailyLimitUp:
    def test_mainboard_limit_up_exact(self) -> None:
        assert detect_daily_limit_up(close_price=11.0, prev_close=10.0, limit_pct=0.10) is True

    def test_mainboard_just_below_limit(self) -> None:
        assert detect_daily_limit_up(close_price=10.99, prev_close=10.0, limit_pct=0.10) is False

    def test_chinext_limit_up(self) -> None:
        assert detect_daily_limit_up(close_price=12.0, prev_close=10.0, limit_pct=0.20) is True

    def test_bse_limit_up(self) -> None:
        assert detect_daily_limit_up(close_price=13.0, prev_close=10.0, limit_pct=0.30) is True

    def test_limit_down_price_is_false(self) -> None:
        assert detect_daily_limit_up(close_price=9.0, prev_close=10.0, limit_pct=0.10) is False

    def test_zero_prev_close_raises(self) -> None:
        with pytest.raises(ValueError, match="prev_close"):
            detect_daily_limit_up(close_price=11.0, prev_close=0, limit_pct=0.10)

    def test_float_tolerance_just_above(self) -> None:
        # 11.0000000001 is above 11.0 - 1e-9 tolerance threshold
        assert (
            detect_daily_limit_up(close_price=11.0000000001, prev_close=10.0, limit_pct=0.10)
            is True
        )

    def test_negative_prev_close_raises(self) -> None:
        with pytest.raises(ValueError, match="prev_close"):
            detect_daily_limit_up(close_price=11.0, prev_close=-1.0, limit_pct=0.10)

    def test_negative_limit_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="limit_pct"):
            detect_daily_limit_up(close_price=11.0, prev_close=10.0, limit_pct=-0.10)


# ---------------------------------------------------------------------------
# detect_daily_limit_down — ≥7 tests
# ---------------------------------------------------------------------------


class TestDetectDailyLimitDown:
    def test_mainboard_limit_down_exact(self) -> None:
        assert detect_daily_limit_down(close_price=9.0, prev_close=10.0, limit_pct=0.10) is True

    def test_mainboard_just_above_limit(self) -> None:
        assert detect_daily_limit_down(close_price=9.01, prev_close=10.0, limit_pct=0.10) is False

    def test_chinext_limit_down(self) -> None:
        assert detect_daily_limit_down(close_price=8.0, prev_close=10.0, limit_pct=0.20) is True

    def test_bse_limit_down(self) -> None:
        assert detect_daily_limit_down(close_price=7.0, prev_close=10.0, limit_pct=0.30) is True

    def test_limit_up_price_is_false(self) -> None:
        assert detect_daily_limit_down(close_price=11.0, prev_close=10.0, limit_pct=0.10) is False

    def test_zero_prev_close_raises(self) -> None:
        with pytest.raises(ValueError, match="prev_close"):
            detect_daily_limit_down(close_price=9.0, prev_close=0, limit_pct=0.10)

    def test_float_tolerance_just_below(self) -> None:
        # 8.9999999999 is below 9.0 + 1e-9 tolerance threshold
        assert (
            detect_daily_limit_down(close_price=8.9999999999, prev_close=10.0, limit_pct=0.10)
            is True
        )

    def test_negative_limit_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="limit_pct"):
            detect_daily_limit_down(close_price=9.0, prev_close=10.0, limit_pct=-0.10)


# ---------------------------------------------------------------------------
# t_plus_n_blocked — ≥7 tests
# ---------------------------------------------------------------------------


class TestTPlusNBlocked:
    def test_same_day_t_plus_1_blocked(self) -> None:
        assert (
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 28),
                t_plus_n=1,
            )
            is True
        )

    def test_next_day_t_plus_1_allowed(self) -> None:
        assert (
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 29),
                t_plus_n=1,
            )
            is False
        )

    def test_t_plus_0_never_blocked(self) -> None:
        assert (
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 28),
                t_plus_n=0,
            )
            is False
        )

    def test_t_plus_2_day_1_blocked(self) -> None:
        assert (
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 29),
                t_plus_n=2,
            )
            is True
        )

    def test_t_plus_2_day_2_allowed(self) -> None:
        assert (
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 30),
                t_plus_n=2,
            )
            is False
        )

    def test_current_before_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="current_date"):
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 27),
                t_plus_n=1,
            )

    def test_negative_t_plus_n_raises(self) -> None:
        with pytest.raises(ValueError, match="t_plus_n"):
            t_plus_n_blocked(
                entry_date=date(2026, 5, 28),
                current_date=date(2026, 5, 29),
                t_plus_n=-1,
            )


# ---------------------------------------------------------------------------
# compute_commission — ≥7 tests
# ---------------------------------------------------------------------------


class TestComputeCommission:
    def test_small_amount_triggers_min_fee(self) -> None:
        # 10000 × 0.0003 = 3.0 < 5.0 → 5.0
        assert compute_commission(trade_amount=10000, rate=0.0003, min_fee=5.0) == 5.0

    def test_large_amount_uses_rate(self) -> None:
        # 100000 × 0.0003 = 30.0 > 5.0 → 30.0
        assert compute_commission(trade_amount=100000, rate=0.0003, min_fee=5.0) == pytest.approx(
            30.0
        )

    def test_zero_min_fee_default(self) -> None:
        # 10000 × 0.0003 = 3.0, min_fee default 0
        assert compute_commission(trade_amount=10000, rate=0.0003) == pytest.approx(3.0)

    def test_zero_trade_amount_with_zero_min_fee(self) -> None:
        assert compute_commission(trade_amount=0, rate=0.0003, min_fee=0.0) == 0.0

    def test_zero_trade_amount_with_min_fee(self) -> None:
        assert compute_commission(trade_amount=0, rate=0.0003, min_fee=5.0) == 5.0

    def test_zero_rate_returns_min_fee(self) -> None:
        assert compute_commission(trade_amount=10000, rate=0.0, min_fee=5.0) == 5.0

    def test_negative_trade_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="trade_amount"):
            compute_commission(trade_amount=-1, rate=0.0003, min_fee=5.0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            compute_commission(trade_amount=10000, rate=-0.0003, min_fee=5.0)

    def test_negative_min_fee_raises(self) -> None:
        with pytest.raises(ValueError, match="min_fee"):
            compute_commission(trade_amount=10000, rate=0.0003, min_fee=-1.0)


# ---------------------------------------------------------------------------
# compute_stamp_tax — ≥8 tests
# ---------------------------------------------------------------------------


class TestComputeStampTax:
    def test_sell_post_cutover_rate(self) -> None:
        # 2023-08-28 以後 rate=0.0005
        assert compute_stamp_tax(trade_amount=10000, rate=0.0005, direction="sell") == 5.0

    def test_sell_pre_cutover_rate(self) -> None:
        # 旧税率 rate=0.001
        assert compute_stamp_tax(trade_amount=10000, rate=0.001, direction="sell") == 10.0

    def test_buy_direction_is_zero(self) -> None:
        assert compute_stamp_tax(trade_amount=10000, rate=0.0005, direction="buy") == 0.0

    def test_both_direction_charges_rate(self) -> None:
        assert compute_stamp_tax(trade_amount=10000, rate=0.0005, direction="both") == 5.0

    def test_large_amount_sell(self) -> None:
        assert compute_stamp_tax(trade_amount=1_000_000, rate=0.0005, direction="sell") == 500.0

    def test_zero_amount(self) -> None:
        assert compute_stamp_tax(trade_amount=0, rate=0.0005, direction="sell") == 0.0

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="trade_amount"):
            compute_stamp_tax(trade_amount=-1, rate=0.0005, direction="sell")

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            compute_stamp_tax(trade_amount=10000, rate=-0.0005, direction="sell")


# ---------------------------------------------------------------------------
# __all__ dual-form import verification
# ---------------------------------------------------------------------------


def test_all_five_in_oprim_all() -> None:
    import oprim

    for sym in (
        "detect_daily_limit_up",
        "detect_daily_limit_down",
        "t_plus_n_blocked",
        "compute_commission",
        "compute_stamp_tax",
    ):
        assert sym in oprim.__all__, f"{sym} missing from oprim.__all__"


def test_top_level_import_form() -> None:
    from oprim import (
        compute_commission,
        compute_stamp_tax,
        detect_daily_limit_down,
        detect_daily_limit_up,
        t_plus_n_blocked,
    )

    assert callable(detect_daily_limit_up)
    assert callable(detect_daily_limit_down)
    assert callable(t_plus_n_blocked)
    assert callable(compute_commission)
    assert callable(compute_stamp_tax)
