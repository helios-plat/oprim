"""Tests for oprim.crypto_scoring — 17 crypto signal scoring primitives."""

from __future__ import annotations

import pytest

from oprim.crypto_scoring import (
    CryptoScoringError,
    score_active_addresses_change,
    score_basis,
    score_cex_balance_change,
    score_etf_inflow,
    score_funding_rate,
    score_lth_change,
    score_ma200_position,
    score_ma50_slope,
    score_ma_arrangement,
    score_max_pain_distance,
    score_mvrv_zscore,
    score_oi_change,
    score_options_skew,
    score_resistance_distance,
    score_stablecoin_inflow,
    score_support_distance,
    score_vpvr_position,
)


# ═══════════════════════════════════════════════════════════════════════════════
# score_ma200_position
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMa200Position:
    def test_at_ma200(self):
        assert score_ma200_position(price=100.0, ma200=100.0) == 0.0

    def test_above_5pct(self):
        assert score_ma200_position(price=105.0, ma200=100.0) == pytest.approx(0.25)

    def test_below_20pct_clamp(self):
        assert score_ma200_position(price=70.0, ma200=100.0) == -1.0

    def test_above_20pct_clamp(self):
        assert score_ma200_position(price=130.0, ma200=100.0) == 1.0

    def test_interpolation_midpoint(self):
        # 10% above → midpoint between (0.05, 0.25) and (0.20, 1.0)
        result = score_ma200_position(price=110.0, ma200=100.0)
        assert -1.0 <= result <= 1.0
        assert result == pytest.approx(0.5, abs=0.01)

    def test_error_on_zero_ma200(self):
        with pytest.raises(CryptoScoringError):
            score_ma200_position(price=100.0, ma200=0.0)

    def test_error_on_negative_ma200(self):
        with pytest.raises(CryptoScoringError):
            score_ma200_position(price=100.0, ma200=-1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# score_ma50_slope
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMa50Slope:
    def test_flat(self):
        assert score_ma50_slope(ma50_now=100.0, ma50_10d_ago=100.0) == 0.0

    def test_positive_5pct(self):
        assert score_ma50_slope(ma50_now=105.0, ma50_10d_ago=100.0) == pytest.approx(0.5)

    def test_clamp_positive(self):
        assert score_ma50_slope(ma50_now=120.0, ma50_10d_ago=100.0) == 1.0

    def test_clamp_negative(self):
        assert score_ma50_slope(ma50_now=80.0, ma50_10d_ago=100.0) == -1.0

    def test_negative_slope(self):
        assert score_ma50_slope(ma50_now=95.0, ma50_10d_ago=100.0) == pytest.approx(-0.5)

    def test_error_on_zero(self):
        with pytest.raises(CryptoScoringError):
            score_ma50_slope(ma50_now=100.0, ma50_10d_ago=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# score_ma_arrangement
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMaArrangement:
    def test_bullish(self):
        assert score_ma_arrangement(price=110.0, ma50=105.0, ma200=100.0) == 1.0

    def test_weak_bullish(self):
        assert score_ma_arrangement(price=110.0, ma50=105.0, ma200=108.0) == 0.5

    def test_weak_bearish(self):
        assert score_ma_arrangement(price=95.0, ma50=100.0, ma200=98.0) == -0.5

    def test_bearish(self):
        assert score_ma_arrangement(price=90.0, ma50=95.0, ma200=100.0) == -1.0

    def test_mixed(self):
        # All equal → no clear arrangement
        assert score_ma_arrangement(price=100.0, ma50=100.0, ma200=100.0) == 0.0

    def test_error_on_zero_price(self):
        with pytest.raises(CryptoScoringError):
            score_ma_arrangement(price=0.0, ma50=100.0, ma200=100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# score_stablecoin_inflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreStablecoinInflow:
    def test_zero_change(self):
        result = score_stablecoin_inflow(change_7d=0.0)
        assert result == pytest.approx(0.0)

    def test_large_positive(self):
        assert score_stablecoin_inflow(change_7d=0.05) == 1.0

    def test_large_negative(self):
        assert score_stablecoin_inflow(change_7d=-0.05) == -1.0

    def test_small_positive(self):
        result = score_stablecoin_inflow(change_7d=0.005)
        assert result == pytest.approx(0.25)

    def test_interpolation(self):
        result = score_stablecoin_inflow(change_7d=0.015)
        assert 0.25 < result < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_etf_inflow
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreEtfInflow:
    def test_zero(self):
        result = score_etf_inflow(net_flow_7d_usd=0.0)
        assert result == pytest.approx(0.0)

    def test_large_positive(self):
        assert score_etf_inflow(net_flow_7d_usd=6e9) == 1.0

    def test_large_negative(self):
        assert score_etf_inflow(net_flow_7d_usd=-6e9) == -1.0

    def test_boundary_positive(self):
        assert score_etf_inflow(net_flow_7d_usd=5e9) == pytest.approx(1.0)

    def test_midpoint(self):
        result = score_etf_inflow(net_flow_7d_usd=0.5e9)
        assert result == pytest.approx(0.25)


# ═══════════════════════════════════════════════════════════════════════════════
# score_cex_balance_change
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreCexBalanceChange:
    def test_zero(self):
        result = score_cex_balance_change(change_7d=0.0)
        assert result == pytest.approx(0.0)

    def test_declining_reserves_bullish(self):
        assert score_cex_balance_change(change_7d=-0.05) == 1.0

    def test_increasing_reserves_bearish(self):
        assert score_cex_balance_change(change_7d=0.05) == -1.0

    def test_small_decline(self):
        assert score_cex_balance_change(change_7d=-0.005) == pytest.approx(0.25)

    def test_clamp_beyond_range(self):
        assert score_cex_balance_change(change_7d=-0.10) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_funding_rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreFundingRate:
    def test_neutral(self):
        assert score_funding_rate(rate_8h=0.0001) == pytest.approx(0.0)

    def test_positive_overheated(self):
        assert score_funding_rate(rate_8h=0.0015) == -1.0

    def test_negative_bullish(self):
        assert score_funding_rate(rate_8h=-0.0015) == 1.0

    def test_clamp_extreme_positive(self):
        assert score_funding_rate(rate_8h=0.01) == -1.0

    def test_interpolation(self):
        result = score_funding_rate(rate_8h=0.0005)
        assert result == pytest.approx(-0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# score_basis
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreBasis:
    def test_zero(self):
        assert score_basis(basis=0.0) == 0.0

    def test_positive_premium(self):
        assert score_basis(basis=0.005) == 1.0

    def test_negative_discount(self):
        assert score_basis(basis=-0.005) == -1.0

    def test_small_positive(self):
        assert score_basis(basis=0.002) == pytest.approx(0.5)

    def test_clamp_extreme(self):
        assert score_basis(basis=0.01) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_mvrv_zscore
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMvrvZscore:
    def test_zero(self):
        assert score_mvrv_zscore(z=0.0) == 0.0

    def test_undervalued(self):
        assert score_mvrv_zscore(z=-1.0) == 1.0

    def test_overvalued(self):
        assert score_mvrv_zscore(z=3.0) == -1.0

    def test_clamp_extreme_low(self):
        assert score_mvrv_zscore(z=-5.0) == 1.0

    def test_midpoint(self):
        result = score_mvrv_zscore(z=1.5)
        assert result == pytest.approx(-0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# score_active_addresses_change
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreActiveAddressesChange:
    def test_zero(self):
        assert score_active_addresses_change(change_7d=0.0) == 0.0

    def test_positive_growth(self):
        assert score_active_addresses_change(change_7d=0.10) == 1.0

    def test_negative_decline(self):
        assert score_active_addresses_change(change_7d=-0.10) == -1.0

    def test_half_growth(self):
        assert score_active_addresses_change(change_7d=0.05) == pytest.approx(0.5)

    def test_clamp_extreme(self):
        assert score_active_addresses_change(change_7d=0.50) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_lth_change
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreLthChange:
    def test_zero(self):
        assert score_lth_change(lth_pct_change=0.0) == 0.0

    def test_positive_hodling(self):
        assert score_lth_change(lth_pct_change=0.02) == 1.0

    def test_negative_distribution(self):
        assert score_lth_change(lth_pct_change=-0.02) == -1.0

    def test_half_positive(self):
        assert score_lth_change(lth_pct_change=0.01) == pytest.approx(0.5)

    def test_clamp_extreme(self):
        assert score_lth_change(lth_pct_change=0.05) == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_options_skew
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreOptionsSkew:
    def test_zero(self):
        assert score_options_skew(skew_pp=0.0) == 0.0

    def test_positive_skew_bearish(self):
        assert score_options_skew(skew_pp=5.0) == -1.0

    def test_negative_skew_bullish(self):
        assert score_options_skew(skew_pp=-5.0) == 1.0

    def test_moderate_positive(self):
        assert score_options_skew(skew_pp=2.0) == pytest.approx(-0.5)

    def test_clamp_extreme(self):
        assert score_options_skew(skew_pp=10.0) == -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_max_pain_distance
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreMaxPainDistance:
    def test_at_max_pain(self):
        assert score_max_pain_distance(distance=0.0) == 0.0

    def test_above_pain_bearish(self):
        assert score_max_pain_distance(distance=0.03) == -1.0

    def test_below_pain_bullish(self):
        assert score_max_pain_distance(distance=-0.03) == 1.0

    def test_small_above(self):
        assert score_max_pain_distance(distance=0.01) == pytest.approx(-0.33)

    def test_clamp_extreme(self):
        assert score_max_pain_distance(distance=0.10) == -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_oi_change
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreOiChange:
    def test_both_below_threshold(self):
        assert score_oi_change(oi_change_7d=0.005, price_change_7d=0.005) == 0.0

    def test_healthy_long_buildup(self):
        assert score_oi_change(oi_change_7d=0.10, price_change_7d=0.05) == pytest.approx(1.0)

    def test_short_buildup(self):
        assert score_oi_change(oi_change_7d=0.10, price_change_7d=-0.05) == pytest.approx(-1.0)

    def test_short_covering(self):
        result = score_oi_change(oi_change_7d=-0.10, price_change_7d=0.05)
        assert result == pytest.approx(0.3)

    def test_long_liquidation(self):
        result = score_oi_change(oi_change_7d=-0.10, price_change_7d=-0.05)
        assert result == pytest.approx(-0.3)

    def test_partial_intensity(self):
        result = score_oi_change(oi_change_7d=0.05, price_change_7d=0.05)
        assert result == pytest.approx(0.5)

    def test_oi_below_threshold_price_above(self):
        assert score_oi_change(oi_change_7d=0.005, price_change_7d=0.10) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_resistance_distance
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreResistanceDistance:
    def test_at_resistance(self):
        assert score_resistance_distance(dist_pct=0.0) == -1.0

    def test_far_from_resistance(self):
        assert score_resistance_distance(dist_pct=5.0) == 0.0

    def test_very_close(self):
        assert score_resistance_distance(dist_pct=1.0) == pytest.approx(-0.7)

    def test_moderate(self):
        assert score_resistance_distance(dist_pct=3.0) == pytest.approx(-0.3)

    def test_beyond_range(self):
        assert score_resistance_distance(dist_pct=10.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_support_distance
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreSupportDistance:
    def test_at_support(self):
        assert score_support_distance(dist_pct=0.0) == 1.0

    def test_far_from_support(self):
        assert score_support_distance(dist_pct=5.0) == 0.0

    def test_very_close(self):
        assert score_support_distance(dist_pct=1.0) == pytest.approx(0.7)

    def test_moderate(self):
        assert score_support_distance(dist_pct=3.0) == pytest.approx(0.3)

    def test_beyond_range(self):
        assert score_support_distance(dist_pct=10.0) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# score_vpvr_position
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoreVpvrPosition:
    def test_low_density_bullish(self):
        assert score_vpvr_position(density_ratio=0.0) == pytest.approx(0.3)

    def test_mid_density_neutral(self):
        assert score_vpvr_position(density_ratio=0.5) == pytest.approx(0.0)

    def test_high_density_bearish(self):
        assert score_vpvr_position(density_ratio=1.0) == pytest.approx(-0.5)

    def test_low_range_flat(self):
        # Between 0.0 and 0.2, score stays at 0.3
        assert score_vpvr_position(density_ratio=0.1) == pytest.approx(0.3)

    def test_high_range_flat(self):
        # Between 0.8 and 1.0, score stays at -0.5
        assert score_vpvr_position(density_ratio=0.9) == pytest.approx(-0.5)
