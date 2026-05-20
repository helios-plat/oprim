"""Tests for oprim.technical.signals — Sprint 0 signal detection primitives."""

from __future__ import annotations

import pytest

from oprim.technical.signals import (
    consecutive_event_count,
    detect_bullish_divergence,
    detect_ma_cross,
    detect_ma_support_bounce,
    detect_price_breakout,
    detect_volume_breakout,
    detect_volume_stagnation,
)


# ── detect_ma_cross ──────────────────────────────────────────────────────────

class TestDetectMaCross:
    def _rising_closes(self, n: int = 30) -> list[float]:
        return [float(100 + i) for i in range(n)]

    def test_golden_cross_detected(self) -> None:
        # Declining then surging: fast crosses above slow at last bar
        # fast=5, slow=10
        # prev (n-1=10): fast_5=avg([7,6,5,6,10])=6.8 < slow_10=avg([15,13,11,9,8,7,6,5,6,10])=9.0 ✓
        # cur (n=11): fast_5=avg([6,5,6,10,30])=11.4 > slow_10=avg([13,11,9,8,7,6,5,6,10,30])=10.5 ✓
        closes = [15.0, 13.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 6.0, 10.0, 30.0]
        result = detect_ma_cross(closes, fast_period=5, slow_period=10, direction="golden")
        assert result is not None
        assert result["crossed"] is True
        assert "fast_ma" in result
        assert "slow_ma" in result
        assert "prev_fast_ma" in result
        assert "prev_slow_ma" in result

    def test_death_cross_detected(self) -> None:
        # Rising then collapsing: fast crosses below slow at last bar
        # prev(n=10): fast_5=avg([11,13,14,10,2])=10.0 >= slow_10=avg([0,2,5,7,9,11,13,14,10,2])=7.3 ✓
        # cur (n=11): fast_5=avg([13,14,10,2,-10])=5.8 < slow_10=avg([2,5,7,9,11,13,14,10,2,-10])=6.3 ✓
        closes = [0.0, 2.0, 5.0, 7.0, 9.0, 11.0, 13.0, 14.0, 10.0, 2.0, -10.0]
        result = detect_ma_cross(closes, fast_period=5, slow_period=10, direction="death")
        assert result is not None
        assert result["crossed"] is True

    def test_returns_none_if_insufficient_data(self) -> None:
        closes = [10.0, 20.0]
        assert detect_ma_cross(closes, 3, 5, "golden") is None

    def test_no_cross_returns_false(self) -> None:
        closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        result = detect_ma_cross(closes, fast_period=2, slow_period=4, direction="golden")
        assert result is not None
        assert result["crossed"] is False

    def test_raises_on_invalid_period(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            detect_ma_cross([1.0, 2.0, 3.0], fast_period=-1, slow_period=5, direction="golden")

    def test_raises_if_fast_ge_slow(self) -> None:
        with pytest.raises(ValueError, match="less than"):
            detect_ma_cross([1.0] * 10, fast_period=5, slow_period=5, direction="golden")

    def test_raises_invalid_direction(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            detect_ma_cross([1.0] * 10, fast_period=2, slow_period=5, direction="sideways")  # type: ignore

    @pytest.mark.academic_reference
    def test_academic_golden_cross_manual(self) -> None:
        """Manual verification: fast=5, slow=10, golden cross at last bar.

        Reference: Murphy (1999), Ch. 9 "Moving Averages".

        Verified manually:
          closes = [15, 13, 11, 9, 8, 7, 6, 5, 6, 10, 30]
          prev (n=10): fast_5 = avg([7,6,5,6,10]) = 6.8
                       slow_10 = avg([15,13,11,9,8,7,6,5,6,10]) = 9.0
                       6.8 < 9.0 → fast below slow ✓
          cur  (n=11): fast_5 = avg([6,5,6,10,30]) = 11.4
                       slow_10 = avg([13,11,9,8,7,6,5,6,10,30]) = 10.5
                       11.4 > 10.5 → fast above slow ✓ → golden cross!
        """
        closes = [15.0, 13.0, 11.0, 9.0, 8.0, 7.0, 6.0, 5.0, 6.0, 10.0, 30.0]
        result = detect_ma_cross(closes, fast_period=5, slow_period=10, direction="golden")
        assert result is not None
        assert result["crossed"] is True
        assert abs(result["prev_fast_ma"] - 6.8) < 1e-9
        assert abs(result["prev_slow_ma"] - 9.0) < 1e-9
        assert abs(result["fast_ma"] - 11.4) < 1e-9
        assert abs(result["slow_ma"] - 10.5) < 1e-9


# ── detect_price_breakout ─────────────────────────────────────────────────────

class TestDetectPriceBreakout:
    def test_breakout_detected(self) -> None:
        highs = [10.0, 11.0, 12.0, 13.0, 14.0]  # max of prior 4 = 13.0
        closes = [9.0, 10.0, 11.0, 12.0, 15.0]  # current close = 15 > 13
        result = detect_price_breakout(highs, closes, window=4)
        assert result is not None
        assert result["broke_out"] is True
        assert result["current_close"] == 15.0
        assert result["prior_max_high"] == 13.0

    def test_no_breakout(self) -> None:
        highs = [10.0, 11.0, 12.0, 13.0, 14.0]
        closes = [9.0, 10.0, 11.0, 12.0, 12.0]
        result = detect_price_breakout(highs, closes, window=4)
        assert result is not None
        assert result["broke_out"] is False

    def test_insufficient_data_returns_none(self) -> None:
        highs = [10.0, 11.0]
        closes = [9.0, 10.0]
        assert detect_price_breakout(highs, closes, window=5) is None

    def test_raises_on_invalid_window(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            detect_price_breakout([1.0, 2.0], [1.0, 2.0], window=0)

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_price_breakout([1.0, 2.0, 3.0], [1.0, 2.0], window=1)

    def test_empty_list_returns_none(self) -> None:
        assert detect_price_breakout([], [], window=5) is None

    @pytest.mark.academic_reference
    def test_academic_donchian_breakout(self) -> None:
        """Verify against Donchian N-day high breakout rule.

        Reference: Donchian, R. (1960). N-day high breakout rule.
        """
        # 20-day channel: if close > 20-day high, breakout
        highs = [float(100 + i % 5) for i in range(21)]
        closes = [float(99 + i % 5) for i in range(21)]
        closes[-1] = 130.0  # definitively above prior 20-day high
        result = detect_price_breakout(highs, closes, window=20)
        assert result is not None
        assert result["broke_out"] is True
        assert result["prior_max_high"] == max(highs[-21:-1])


# ── detect_volume_breakout ────────────────────────────────────────────────────

class TestDetectVolumeBreakout:
    def test_volume_breakout_detected(self) -> None:
        volumes = [100.0, 100.0, 100.0, 500.0]  # last is 5× prior avg
        closes = [10.0, 10.0, 10.0, 15.0]        # above prior max high
        highs = [10.0, 11.0, 10.0, 14.0]
        result = detect_volume_breakout(volumes, closes, highs, vol_ratio_threshold=2.0)
        assert result is not None
        assert result["broke_out"] is True
        assert abs(result["vol_ratio"] - 500.0 / (300.0 / 3)) < 1e-9

    def test_no_breakout_low_volume(self) -> None:
        volumes = [100.0, 100.0, 100.0, 110.0]  # not 2× ratio
        closes = [10.0, 10.0, 10.0, 15.0]
        highs = [10.0, 11.0, 10.0, 14.0]
        result = detect_volume_breakout(volumes, closes, highs)
        assert result is not None
        assert result["broke_out"] is False

    def test_insufficient_data_returns_none(self) -> None:
        assert detect_volume_breakout([100.0], [10.0], [10.0]) is None

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_volume_breakout([1.0, 2.0], [1.0], [1.0, 2.0])

    @pytest.mark.academic_reference
    def test_academic_vol_ratio_calculation(self) -> None:
        """Verify vol_ratio = current_vol / mean(prior_vols).

        Reference: Pring (2014), Ch. 13 "Volume Confirmation".
        """
        volumes = [50.0, 60.0, 70.0, 300.0]  # prior avg = 60
        closes = [10.0, 11.0, 12.0, 20.0]
        highs = [10.0, 11.0, 12.0, 19.0]
        result = detect_volume_breakout(volumes, closes, highs, vol_ratio_threshold=2.0)
        assert result is not None
        expected_ratio = 300.0 / ((50.0 + 60.0 + 70.0) / 3)
        assert abs(result["vol_ratio"] - expected_ratio) < 1e-9


# ── detect_ma_support_bounce ──────────────────────────────────────────────────

class TestDetectMaSupportBounce:
    def test_bounce_detected(self) -> None:
        # MA of prior 3 closes = (10+10+10)/3 = 10.0
        opens =   [10.0, 10.0, 10.0, 10.0, 9.0]
        lows =    [10.0, 10.0, 10.0, 10.0, 9.95]  # within 1.5% of MA=10
        closes =  [10.0, 10.0, 10.0, 10.0, 10.5]  # closes > open ✓
        volumes = [100.0, 100.0, 100.0, 100.0, 200.0]  # vol higher ✓
        result = detect_ma_support_bounce(opens, lows, closes, volumes, ma_period=3)
        assert result is not None
        assert result["bounced"] is True
        assert abs(result["ma_value"] - 10.0) < 1e-9
        assert result["touch_low"] == 9.95

    def test_no_bounce_low_far_from_ma(self) -> None:
        opens =   [10.0, 10.0, 10.0, 10.0, 9.0]
        lows =    [10.0, 10.0, 10.0, 10.0, 5.0]  # way below MA
        closes =  [10.0, 10.0, 10.0, 10.0, 9.5]
        volumes = [100.0, 100.0, 100.0, 100.0, 200.0]
        result = detect_ma_support_bounce(opens, lows, closes, volumes, ma_period=3)
        assert result is not None
        assert result["bounced"] is False

    def test_insufficient_data_returns_none(self) -> None:
        assert detect_ma_support_bounce(
            [10.0], [10.0], [10.0], [100.0], ma_period=5
        ) is None

    def test_raises_on_invalid_ma_period(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            detect_ma_support_bounce([1.0] * 5, [1.0] * 5, [1.0] * 5, [1.0] * 5, ma_period=0)

    @pytest.mark.academic_reference
    def test_academic_ma_support_tolerance(self) -> None:
        """Verify tolerance_pct=1.5% MA touch detection.

        Reference: Murphy (1999), Ch. 4 "Trends and Trendlines".
        """
        # MA = 100.0, tolerance_pct = 0.015 → touch range [98.5, 101.5]
        opens =   [100.0, 100.0, 100.0, 100.0, 98.0]
        lows =    [100.0, 100.0, 100.0, 100.0, 99.0]  # 99 within 1.5% of 100 ✓
        closes =  [100.0, 100.0, 100.0, 100.0, 101.0]  # closes > open ✓
        volumes = [100.0, 100.0, 100.0, 100.0, 150.0]
        result = detect_ma_support_bounce(opens, lows, closes, volumes, ma_period=3, tolerance_pct=0.015)
        assert result is not None
        assert abs(result["ma_value"] - 100.0) < 1e-9
        assert result["bounced"] is True


# ── detect_volume_stagnation ──────────────────────────────────────────────────

class TestDetectVolumeStagnation:
    def test_stagnation_detected(self) -> None:
        opens =   [10.0, 10.0, 10.0, 10.0]
        highs =   [10.0, 10.0, 10.0, 15.0]  # long upper shadow
        closes =  [10.0, 10.0, 10.0, 10.1]  # tiny body
        volumes = [100.0, 100.0, 100.0, 400.0]  # 4× = high volume
        result = detect_volume_stagnation(opens, highs, closes, volumes, vol_ratio_threshold=2.0)
        assert result is not None
        assert result["stagnated"] is True
        assert result["vol_ratio"] == pytest.approx(400.0 / 100.0, rel=1e-6)

    def test_no_stagnation_low_volume(self) -> None:
        opens =   [10.0, 10.0, 10.0, 10.0]
        highs =   [10.0, 10.0, 10.0, 11.0]
        closes =  [10.0, 10.0, 10.0, 10.1]
        volumes = [100.0, 100.0, 100.0, 110.0]  # not 2×
        result = detect_volume_stagnation(opens, highs, closes, volumes)
        assert result is not None
        assert result["stagnated"] is False

    def test_insufficient_data_returns_none(self) -> None:
        assert detect_volume_stagnation([10.0], [10.0], [10.0], [100.0]) is None

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_volume_stagnation([1.0, 2.0], [1.0, 2.0], [1.0], [1.0, 2.0])

    @pytest.mark.academic_reference
    def test_academic_body_ratio(self) -> None:
        """Verify body_ratio = |close-open| / (high - min(open,close)).

        Reference: Nison (1991). Japanese Candlestick Charting Techniques.
        """
        opens =   [10.0, 10.0]
        highs =   [10.0, 20.0]  # 10-point range from base
        closes =  [10.0, 10.5]  # 0.5-point body
        volumes = [100.0, 1000.0]
        result = detect_volume_stagnation(opens, highs, closes, volumes, vol_ratio_threshold=2.0)
        assert result is not None
        # body = |10.5 - 10.0| = 0.5, base = min(10.0, 10.5) = 10.0, span = 20 - 10 = 10
        # body_ratio = 0.5 / 10 = 0.05
        assert abs(result["body_ratio"] - 0.05) < 1e-9
        assert result["stagnated"] is True  # vol 1000 / 100 = 10 >= 2


# ── detect_bullish_divergence ─────────────────────────────────────────────────

class TestDetectBullishDivergence:
    def test_divergence_detected(self) -> None:
        # price: lower low (10 → 8), indicator: higher low (20 → 25)
        prices =    [15.0, 10.0, 15.0, 12.0, 8.0, 12.0]
        indicator = [30.0, 20.0, 30.0, 28.0, 25.0, 28.0]
        result = detect_bullish_divergence(prices, indicator)
        assert result is not None
        assert result["diverged"] is True

    def test_no_divergence(self) -> None:
        # both price and indicator make lower lows
        prices =    [15.0, 10.0, 15.0, 12.0, 8.0, 12.0]
        indicator = [30.0, 20.0, 30.0, 28.0, 15.0, 28.0]
        result = detect_bullish_divergence(prices, indicator)
        assert result is not None
        assert result["diverged"] is False

    def test_insufficient_data_returns_none(self) -> None:
        assert detect_bullish_divergence([10.0, 9.0, 8.0], [1.0, 2.0, 3.0]) is None

    def test_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            detect_bullish_divergence([1.0, 2.0, 3.0, 4.0], [1.0, 2.0])

    @pytest.mark.academic_reference
    def test_academic_divergence_definition(self) -> None:
        """Verify: lower price low + higher indicator low = bullish divergence.

        Reference: Murphy (1999), Ch. 10 "Oscillators and Divergences".
        """
        prices =    [20.0, 15.0, 20.0, 18.0, 12.0, 18.0]
        indicator = [50.0, 30.0, 50.0, 48.0, 35.0, 48.0]
        result = detect_bullish_divergence(prices, indicator)
        assert result is not None
        # first half min: price=15 at idx=1, ind=30
        # second half min: price=12 at idx=4 (second half idx=1), ind=35
        assert result["price_low_1"] == 15.0
        assert result["price_low_2"] == 12.0
        assert result["indicator_low_1"] == 30.0
        assert result["indicator_low_2"] == 35.0
        assert result["diverged"] is True


# ── consecutive_event_count ───────────────────────────────────────────────────

class TestConsecutiveEventCount:
    def test_all_true(self) -> None:
        assert consecutive_event_count([True, True, True]) == 3

    def test_trailing_true(self) -> None:
        assert consecutive_event_count([True, False, True, True, True]) == 3

    def test_ends_false(self) -> None:
        assert consecutive_event_count([True, True, False]) == 0

    def test_empty_list(self) -> None:
        assert consecutive_event_count([]) == 0

    def test_single_true(self) -> None:
        assert consecutive_event_count([True]) == 1

    def test_single_false(self) -> None:
        assert consecutive_event_count([False]) == 0

    def test_all_false(self) -> None:
        assert consecutive_event_count([False, False, False]) == 0

    @pytest.mark.academic_reference
    def test_academic_run_length(self) -> None:
        """Verify tail run-length counting per standard algorithm.

        Reference: standard run-length encoding.
        """
        events = [True, False, True, True, True, False, True, True]
        # Last run of True: length 2
        assert consecutive_event_count(events) == 2
