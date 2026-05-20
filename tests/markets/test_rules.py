"""Tests for oprim.markets.rules — Sprint 0 market regulatory rule primitives."""

from __future__ import annotations

from datetime import date

import pytest

from oprim.markets.rules import commission, stamp_tax, t_plus_n_blocked


class TestStampTax:
    def test_sell_side_tax(self) -> None:
        # A-share post-2023-08-28: 0.0005 on sell
        result = stamp_tax(10_000.0, 0.0005, "sell")
        assert result == pytest.approx(5.0)

    def test_buy_side_tax(self) -> None:
        result = stamp_tax(10_000.0, 0.001, "buy")
        assert result == pytest.approx(10.0)

    def test_both_sides_tax(self) -> None:
        result = stamp_tax(10_000.0, 0.001, "both")
        assert result == pytest.approx(10.0)

    def test_zero_amount(self) -> None:
        assert stamp_tax(0.0, 0.001, "sell") == 0.0

    def test_negative_amount(self) -> None:
        assert stamp_tax(-100.0, 0.001, "sell") == 0.0

    def test_zero_rate(self) -> None:
        assert stamp_tax(10_000.0, 0.0, "sell") == 0.0

    def test_raises_on_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            stamp_tax(10_000.0, -0.001, "sell")

    def test_raises_on_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            stamp_tax(10_000.0, 0.001, "both_sides")  # type: ignore

    @pytest.mark.academic_reference
    def test_academic_a_share_stamp_tax(self) -> None:
        """Verify A-share stamp tax: 0.0005 on sell side.

        Reference: CSRC Notice, August 28, 2023 (税率从0.1%降至0.05%).
        """
        trade_amount = 100_000.0
        rate = 0.0005  # post-2023-08-28 A-share sell-only rate
        expected_tax = 50.0
        assert stamp_tax(trade_amount, rate, "sell") == pytest.approx(expected_tax)


class TestTPlusNBlocked:
    def test_t_plus_1_same_day_blocked(self) -> None:
        # Buy and sell on same day: 0 business days elapsed < 1 → blocked
        assert t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 19), n=1) is True

    def test_t_plus_1_next_day_not_blocked(self) -> None:
        # Buy Tue, sell Wed: 1 business day elapsed >= 1 → not blocked
        assert t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 20), n=1) is False

    def test_t_plus_0_same_day_not_blocked(self) -> None:
        # T+0: sell on same day allowed
        assert t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 19), n=0) is False

    def test_t_plus_2_next_day_blocked(self) -> None:
        # T+2: 1 business day is not enough
        assert t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 20), n=2) is True

    def test_t_plus_2_two_days_later_not_blocked(self) -> None:
        # Buy Tue, sell Thu: 2 business days elapsed >= 2 → not blocked
        assert t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 21), n=2) is False

    def test_sell_date_before_buy_date_blocked(self) -> None:
        # 0 elapsed business days
        assert t_plus_n_blocked(date(2026, 5, 20), date(2026, 5, 19), n=1) is True

    def test_spans_weekend(self) -> None:
        # Buy Friday, sell Monday: 1 business day elapsed → not blocked for T+1
        assert t_plus_n_blocked(date(2026, 5, 15), date(2026, 5, 18), n=1) is False

    def test_raises_on_negative_n(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            t_plus_n_blocked(date(2026, 5, 19), date(2026, 5, 20), n=-1)

    @pytest.mark.academic_reference
    def test_academic_t_plus_1_rule(self) -> None:
        """Verify A-share T+1 rule: must hold overnight before selling.

        Reference: CSRC T+1 settlement rule for A-share equity.
        """
        buy_date = date(2026, 5, 19)  # Tuesday
        # Cannot sell on same day
        assert t_plus_n_blocked(buy_date, date(2026, 5, 19), n=1) is True
        # Can sell next business day
        assert t_plus_n_blocked(buy_date, date(2026, 5, 20), n=1) is False
        # Friday buy → Monday sell: 1 business day (Mon=1)
        assert t_plus_n_blocked(date(2026, 5, 15), date(2026, 5, 18), n=1) is False


class TestCommission:
    def test_typical_rate_no_min(self) -> None:
        result = commission(10_000.0, 0.00025)
        assert result == pytest.approx(2.5)

    def test_min_fee_applied(self) -> None:
        # Small trade: 10 * 0.00025 = 0.0025 < min_fee=5.0
        result = commission(10.0, 0.00025, min_fee=5.0)
        assert result == pytest.approx(5.0)

    def test_rate_exceeds_min_fee(self) -> None:
        # 100_000 * 0.00025 = 25.0 > min_fee=5.0
        result = commission(100_000.0, 0.00025, min_fee=5.0)
        assert result == pytest.approx(25.0)

    def test_zero_amount(self) -> None:
        assert commission(0.0, 0.001) == 0.0

    def test_negative_amount(self) -> None:
        assert commission(-500.0, 0.001) == 0.0

    def test_zero_rate_with_min_fee(self) -> None:
        result = commission(10_000.0, 0.0, min_fee=5.0)
        assert result == pytest.approx(5.0)

    def test_zero_rate_no_min_fee(self) -> None:
        assert commission(10_000.0, 0.0) == 0.0

    def test_raises_on_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            commission(10_000.0, -0.001)

    def test_raises_on_negative_min_fee(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            commission(10_000.0, 0.001, min_fee=-1.0)

    @pytest.mark.academic_reference
    def test_academic_a_share_commission(self) -> None:
        """Verify max(amount * rate, min_fee) formula.

        Reference: A-share retail brokerage commission (typical: 0.025%, min 5 CNY).
        """
        rate = 0.00025
        min_fee = 5.0
        # Small trade
        assert commission(1_000.0, rate, min_fee) == pytest.approx(5.0)
        # Large trade: 50_000 * 0.00025 = 12.5 > 5
        assert commission(50_000.0, rate, min_fee) == pytest.approx(12.5)
        # Boundary: exactly 20_000 * 0.00025 = 5.0 = min_fee
        assert commission(20_000.0, rate, min_fee) == pytest.approx(5.0)
