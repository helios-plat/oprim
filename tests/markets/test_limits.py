"""Tests for oprim.markets.limits — Sprint 0 daily price limit primitives."""

from __future__ import annotations

import pytest

from oprim.markets.limits import detect_daily_limit_down, detect_daily_limit_up, seal_strength


class TestDetectDailyLimitUp:
    def test_a_share_main_board_hit(self) -> None:
        assert detect_daily_limit_up(11.0, 10.0, 0.10) is True

    def test_a_share_chinext_hit(self) -> None:
        assert detect_daily_limit_up(12.0, 10.0, 0.20) is True

    def test_not_at_limit(self) -> None:
        assert detect_daily_limit_up(10.5, 10.0, 0.10) is False

    def test_slightly_below_limit_with_tolerance(self) -> None:
        # limit_price = 11.0, tolerance=0.0005 → threshold = 10.9995
        assert detect_daily_limit_up(10.9996, 10.0, 0.10) is True

    def test_exactly_at_limit(self) -> None:
        assert detect_daily_limit_up(11.0, 10.0, 0.10) is True

    def test_raises_on_zero_prev_close(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            detect_daily_limit_up(1.0, 0.0, 0.10)

    def test_raises_on_negative_limit_pct(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            detect_daily_limit_up(11.0, 10.0, -0.1)

    def test_zero_limit_pct(self) -> None:
        # No limit (limit_pct=0): any close >= prev_close - tolerance triggers
        assert detect_daily_limit_up(10.0, 10.0, 0.0) is True

    @pytest.mark.academic_reference
    def test_academic_formula_verification(self) -> None:
        """Verify formula: close >= prev_close * (1 + limit_pct) - tolerance.

        Reference: A-share daily price limit rule, CSRC regulation.
        """
        prev_close = 15.32
        limit_pct = 0.10
        tolerance = 0.0005
        limit_price = prev_close * (1 + limit_pct)
        # Exactly at limit
        assert detect_daily_limit_up(limit_price, prev_close, limit_pct, tolerance) is True
        # Just below tolerance
        assert detect_daily_limit_up(limit_price - 0.001, prev_close, limit_pct, tolerance) is False
        # Within tolerance
        assert detect_daily_limit_up(limit_price - 0.0001, prev_close, limit_pct, tolerance) is True


class TestDetectDailyLimitDown:
    def test_a_share_limit_down(self) -> None:
        assert detect_daily_limit_down(9.0, 10.0, 0.10) is True

    def test_not_at_lower_limit(self) -> None:
        assert detect_daily_limit_down(9.5, 10.0, 0.10) is False

    def test_exactly_at_lower_limit(self) -> None:
        assert detect_daily_limit_down(9.0, 10.0, 0.10) is True

    def test_slightly_above_limit_with_tolerance(self) -> None:
        # limit_price = 9.0, tolerance=0.0005 → threshold = 9.0005
        assert detect_daily_limit_down(9.0004, 10.0, 0.10) is True

    def test_raises_on_negative_prev_close(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            detect_daily_limit_down(5.0, -1.0, 0.10)

    @pytest.mark.academic_reference
    def test_academic_symmetric_to_limit_up(self) -> None:
        """Verify limit_down is symmetric to limit_up.

        Reference: A-share daily limit rules (CSRC).
        """
        prev_close = 10.0
        limit_pct = 0.10
        limit_price_down = prev_close * (1 - limit_pct)  # 9.0
        assert detect_daily_limit_down(limit_price_down, prev_close, limit_pct) is True
        assert detect_daily_limit_down(limit_price_down + 0.01, prev_close, limit_pct) is False


class TestSealStrength:
    def test_typical_ratio(self) -> None:
        result = seal_strength(500_000.0, 1_000_000.0)
        assert result == pytest.approx(0.5)

    def test_zero_seal(self) -> None:
        assert seal_strength(0.0, 1_000_000.0) == 0.0

    def test_zero_volume_returns_zero(self) -> None:
        assert seal_strength(500_000.0, 0.0) == 0.0

    def test_negative_seal_returns_zero(self) -> None:
        assert seal_strength(-1.0, 1_000_000.0) == 0.0

    def test_seal_exceeds_volume_allowed(self) -> None:
        # seal can be > volume in extreme cases (e.g., large queued orders)
        result = seal_strength(2_000_000.0, 1_000_000.0)
        assert result == pytest.approx(2.0)

    @pytest.mark.academic_reference
    def test_academic_order_book_imbalance(self) -> None:
        """Verify seal_strength = seal_amount / total_volume.

        Reference: market microstructure order book imbalance literature.
        """
        # Example from limit-up board monitoring:
        # seal amount = 1.2 billion, total volume = 500 million
        seal = 1_200_000_000.0
        volume = 500_000_000.0
        assert seal_strength(seal, volume) == pytest.approx(2.4, rel=1e-9)
