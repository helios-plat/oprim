"""Tests for crypto_technical oprims."""
import pytest

from oprim.crypto_technical import (
    CryptoTechnicalError,
    compute_cross_asset_divergence_revert,
    compute_stablecoin_event_revert,
    compute_vpvr,
    detect_pivots,
)


class TestComputeVpvr:
    def test_single_bar(self):
        klines = [{"low": 90, "high": 110, "volume": 1000}]
        result = compute_vpvr(klines=klines, spot=100.0)
        assert result is not None
        spot_d, max_d = result
        assert spot_d > 0
        assert max_d > 0

    def test_spot_at_high_volume(self):
        klines = [
            {"low": 90, "high": 100, "volume": 5000},
            {"low": 100, "high": 110, "volume": 100},
        ]
        result = compute_vpvr(klines=klines, spot=95.0)
        assert result is not None
        spot_d, max_d = result
        assert spot_d == max_d  # spot is in the high-volume zone

    def test_empty_klines(self):
        assert compute_vpvr(klines=[], spot=100.0) is None

    def test_spot_out_of_range(self):
        klines = [{"low": 90, "high": 110, "volume": 1000}]
        assert compute_vpvr(klines=klines, spot=50.0) is None

    def test_invalid_spot(self):
        klines = [{"low": 90, "high": 110, "volume": 1000}]
        assert compute_vpvr(klines=klines, spot=-1.0) is None

    def test_custom_buckets(self):
        klines = [{"low": 100, "high": 200, "volume": 1000}]
        result = compute_vpvr(klines=klines, spot=150.0, n_buckets=10)
        assert result is not None


class TestDetectPivots:
    def test_basic(self):
        support, resistance = detect_pivots(highs=[110, 105, 108], lows=[90, 92, 91])
        assert support == 90.0
        assert resistance == 110.0

    def test_lookback_limits(self):
        highs = [100, 110, 105, 108, 120]
        lows = [80, 85, 82, 88, 90]
        support, resistance = detect_pivots(highs=highs, lows=lows, lookback=2)
        assert resistance == 120.0
        assert support == 88.0

    def test_single_element(self):
        support, resistance = detect_pivots(highs=[100.0], lows=[90.0])
        assert support == 90.0
        assert resistance == 100.0

    def test_empty_raises(self):
        with pytest.raises(CryptoTechnicalError):
            detect_pivots(highs=[], lows=[90])

    def test_empty_lows_raises(self):
        with pytest.raises(CryptoTechnicalError):
            detect_pivots(highs=[100], lows=[])


class TestCrossAssetDivergenceRevert:
    def test_revert_bullish(self):
        # BTC +20%, ETH -6.7%, SOL -5% → spread = 26.7% > 20%, SOL weakest
        r = compute_cross_asset_divergence_revert(
            btc_close_30d_ago=50000, btc_close_now=60000,
            eth_close_30d_ago=3000, eth_close_now=2800,
            sol_close_30d_ago=100, sol_close_now=80,
            target="SOL-USDT",
        )
        assert r["available"] is True
        assert r["value"] == 1.0
        assert r["signal"] == "revert_bullish"

    def test_spread_below_threshold(self):
        r = compute_cross_asset_divergence_revert(
            btc_close_30d_ago=100, btc_close_now=105,
            eth_close_30d_ago=100, eth_close_now=104,
            sol_close_30d_ago=100, sol_close_now=103,
            target="SOL-USDT",
        )
        assert r["available"] is False
        assert r["signal"] == "spread_below_threshold"

    def test_target_not_weakest(self):
        r = compute_cross_asset_divergence_revert(
            btc_close_30d_ago=50000, btc_close_now=60000,
            eth_close_30d_ago=3000, eth_close_now=2800,
            sol_close_30d_ago=100, sol_close_now=130,
            target="SOL-USDT",
        )
        assert r["available"] is False
        assert r["signal"] == "target_not_weakest"

    def test_invalid_prices(self):
        r = compute_cross_asset_divergence_revert(
            btc_close_30d_ago=0, btc_close_now=60000,
            eth_close_30d_ago=3000, eth_close_now=2800,
            sol_close_30d_ago=100, sol_close_now=95,
            target="ETH-USDT",
        )
        assert r["available"] is False
        assert r["signal"] == "invalid_prices"

    def test_negative_price(self):
        r = compute_cross_asset_divergence_revert(
            btc_close_30d_ago=-1, btc_close_now=60000,
            eth_close_30d_ago=3000, eth_close_now=2800,
            sol_close_30d_ago=100, sol_close_now=95,
            target="ETH-USDT",
        )
        assert r["available"] is False


class TestStablecoinEventRevert:
    def test_large_burn(self):
        r = compute_stablecoin_event_revert(net_mint_burn_24h=-600_000_000)
        assert r["available"] is True
        assert r["value"] == -1.0
        assert r["tier"] == "$500M+"

    def test_billion_burn(self):
        r = compute_stablecoin_event_revert(net_mint_burn_24h=-1_500_000_000)
        assert r["tier"] == "$1B+"

    def test_small_burn_no_signal(self):
        r = compute_stablecoin_event_revert(net_mint_burn_24h=-100_000_000)
        assert r["available"] is False

    def test_mint_no_signal(self):
        r = compute_stablecoin_event_revert(net_mint_burn_24h=800_000_000)
        assert r["available"] is False

    def test_none_input(self):
        r = compute_stablecoin_event_revert(net_mint_burn_24h=None)
        assert r["available"] is False
        assert r["signal"] == "no_data"
