"""Tests for crypto_lookup oprims."""
import pytest

from oprim.crypto_lookup import (
    CryptoLookupError,
    regime_score,
    seasonality_score,
    sector_rotation_score,
)


class TestRegimeScore:
    def test_bull_low_vol(self):
        r = regime_score(regime="bull_low_vol", confidence=0.8)
        assert r["value"] == 40
        assert r["confidence"] == 0.8

    def test_bear_high_vol(self):
        r = regime_score(regime="bear_high_vol", confidence=0.9)
        assert r["value"] == -40

    def test_unknown_regime(self):
        r = regime_score(regime="unknown")
        assert r["value"] == 0
        assert r["confidence"] == 0.5

    def test_unrecognized_regime(self):
        r = regime_score(regime="nonsense")
        assert r["value"] == 0
        assert r["confidence"] == 0.1

    def test_invalid_confidence_raises(self):
        with pytest.raises(CryptoLookupError):
            regime_score(regime="bull_low_vol", confidence=1.5)


class TestSeasonalityScore:
    def test_october_btc(self):
        r = seasonality_score(month=10, symbol="BTC-USDT")
        assert r["value"] == 100  # 15/15*100 = 100

    def test_may_btc(self):
        r = seasonality_score(month=5, symbol="BTC-USDT")
        assert r["value"] < 0

    def test_halving_bonus(self):
        r = seasonality_score(month=1, symbol="BTC-USDT", months_since_halving=12.0)
        assert r["value"] > seasonality_score(month=1, symbol="BTC-USDT")["value"]

    def test_halving_no_bonus_outside_window(self):
        r1 = seasonality_score(month=1, symbol="BTC-USDT", months_since_halving=3.0)
        r2 = seasonality_score(month=1, symbol="BTC-USDT")
        assert r1["value"] == r2["value"]

    def test_eth_symbol(self):
        r = seasonality_score(month=1, symbol="ETH-USDT")
        assert r["value"] == 100  # 15/15*100

    def test_invalid_month_raises(self):
        with pytest.raises(CryptoLookupError):
            seasonality_score(month=0)

    def test_month_13_raises(self):
        with pytest.raises(CryptoLookupError):
            seasonality_score(month=13)


class TestSectorRotationScore:
    def test_high_btc_dom_rising(self):
        r = sector_rotation_score(btc_dominance=62.0, btc_dom_change=0.5)
        assert r["value"] == 30

    def test_low_btc_dom(self):
        r = sector_rotation_score(btc_dominance=45.0)
        assert r["value"] == -10

    def test_eth_btc_rising(self):
        r = sector_rotation_score(eth_btc_change=0.05)
        assert r["value"] == 10

    def test_both_indicators(self):
        r = sector_rotation_score(btc_dominance=62.0, eth_btc_change=0.05)
        assert r["value"] == 40

    def test_no_data(self):
        r = sector_rotation_score()
        assert r["value"] == 0
        assert r["confidence"] == 0.0

    def test_neutral_dom(self):
        r = sector_rotation_score(btc_dominance=55.0)
        assert r["value"] == 0
        assert "neutral" in r["contributors"][0]
