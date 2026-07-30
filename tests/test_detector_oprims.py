"""Tests for B9 realtime detector oprims (7 detectors, ≥5 tests each)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from oprim._detector_types import DetectorSignal
from oprim._exceptions import OprimError
from oprim.detect_sector_collapse import SectorCollapseConfig, detect_sector_collapse
from oprim.detect_dragon_switch import DragonSwitchConfig, detect_dragon_switch
from oprim.detect_hot_money_converge import HotMoneyConvergeConfig, detect_hot_money_converge
from oprim.detect_limit_board_explosion import (
    LimitBoardExplosionConfig,
    detect_limit_board_explosion,
)
from oprim.detect_volume_spike import VolumeSpikeConfig, detect_volume_spike
from oprim.detect_northbound_reversal import NorthboundReversalConfig, detect_northbound_reversal
from oprim.detect_news_shock import NewsShockConfig, detect_news_shock
from oprim.financial_metric_extraction import FinancialMetric, NewsItem


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_signal(result) -> bool:
    return isinstance(result, DetectorSignal)


# ── 1. detect_sector_collapse ─────────────────────────────────────────────────


class TestDetectSectorCollapse:
    def test_hit_returns_signal(self):
        sig = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=96.0,
            constituent_changes=[-0.08, 0.05, -0.07, 0.09, -0.04],  # std≈0.076 > 0.05
        )
        assert _is_signal(sig)
        assert sig.detector_name == "sector_collapse"

    def test_miss_when_drop_insufficient(self):
        result = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=99.0,
            constituent_changes=[-0.05, 0.04, -0.03, 0.06],
        )
        assert result is None

    def test_miss_when_std_below_threshold(self):
        result = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=96.0,
            constituent_changes=[-0.03, -0.03, -0.03, -0.03],  # uniform → low std
        )
        assert result is None

    def test_invalid_price_raises(self):
        with pytest.raises(OprimError, match="price_1h_ago"):
            detect_sector_collapse(
                price_1h_ago=0.0,
                price_now=96.0,
                constituent_changes=[-0.05, 0.04],
            )

    def test_too_few_constituents_raises(self):
        with pytest.raises(OprimError, match="constituent_changes"):
            detect_sector_collapse(
                price_1h_ago=100.0,
                price_now=96.0,
                constituent_changes=[-0.05],
            )

    def test_evidence_keys_present(self):
        sig = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=96.0,
            constituent_changes=[-0.08, 0.05, -0.07, 0.09, -0.04],  # std≈0.076 > 0.05
        )
        assert sig is not None
        assert "drop_1h" in sig.evidence
        assert "divergence_std" in sig.evidence

    def test_triggered_at_is_datetime(self):
        sig = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=96.0,
            constituent_changes=[-0.08, 0.05, -0.07, 0.09],  # std≈0.085 > 0.05
        )
        assert sig is not None
        assert isinstance(sig.triggered_at, datetime)

    def test_critical_on_deep_drop(self):
        cfg = SectorCollapseConfig(drop_threshold_1h=-0.02, divergence_std_threshold=0.03)
        sig = detect_sector_collapse(
            price_1h_ago=100.0,
            price_now=92.0,  # -8%, well below 1.5x threshold (-3%)
            constituent_changes=[-0.10, 0.05, -0.08, 0.07, -0.09],
            config=cfg,
        )
        assert sig is not None
        assert sig.severity == "critical"


# ── 2. detect_dragon_switch ───────────────────────────────────────────────────


class TestDetectDragonSwitch:
    def test_hit_returns_signal(self):
        sig = detect_dragon_switch(
            top1_change_pct=-0.02,
            new_top3_vol_ratios=[3.1, 2.8, 3.5],
        )
        assert _is_signal(sig)
        assert sig.detector_name == "dragon_switch"

    def test_miss_when_top1_positive(self):
        result = detect_dragon_switch(
            top1_change_pct=0.03,
            new_top3_vol_ratios=[3.1, 2.8, 3.5],
        )
        assert result is None

    def test_miss_when_new_vol_too_low(self):
        result = detect_dragon_switch(
            top1_change_pct=-0.02,
            new_top3_vol_ratios=[1.2, 1.0, 0.9],
        )
        assert result is None

    def test_empty_vol_ratios_raises(self):
        with pytest.raises(OprimError):
            detect_dragon_switch(top1_change_pct=-0.02, new_top3_vol_ratios=[])

    def test_min_leaders_threshold_enforced(self):
        cfg = DragonSwitchConfig(
            new_top3_vol_ratio_threshold=2.0, min_new_leaders_above_threshold=3
        )
        result = detect_dragon_switch(
            top1_change_pct=-0.02,
            new_top3_vol_ratios=[3.0, 3.0, 1.0],  # only 2 above threshold
            config=cfg,
        )
        assert result is None

    def test_high_severity_on_large_vol(self):
        sig = detect_dragon_switch(
            top1_change_pct=-0.03,
            new_top3_vol_ratios=[5.0, 6.0],
        )
        assert sig is not None
        assert sig.severity == "high"

    def test_evidence_contains_vol_ratios(self):
        sig = detect_dragon_switch(
            top1_change_pct=-0.02,
            new_top3_vol_ratios=[3.1, 2.8],
        )
        assert sig is not None
        assert "new_top3_vol_ratios" in sig.evidence
        assert sig.evidence["top1_change_pct"] == pytest.approx(-0.02)


# ── 3. detect_hot_money_converge ──────────────────────────────────────────────


_KNOWN_SEATS = ["方正证券成都营业部", "国泰君安上海分公司", "华泰证券南京分公司"]


class TestDetectHotMoneyConverge:
    def test_hit_returns_signal(self):
        sig = detect_hot_money_converge(
            seat_names=["方正证券成都营业部", "随机券商X", "国泰君安上海分公司"],
            net_buy_total=8000.0,
            known_tycoon_seats=_KNOWN_SEATS,
        )
        assert _is_signal(sig)
        assert sig.detector_name == "hot_money_converge"

    def test_miss_when_too_few_hits(self):
        result = detect_hot_money_converge(
            seat_names=["方正证券成都营业部", "随机券商X"],
            net_buy_total=8000.0,
            known_tycoon_seats=_KNOWN_SEATS,
        )
        assert result is None

    def test_miss_when_net_buy_too_low(self):
        result = detect_hot_money_converge(
            seat_names=["方正证券成都营业部", "国泰君安上海分公司"],
            net_buy_total=100.0,
            known_tycoon_seats=_KNOWN_SEATS,
        )
        assert result is None

    def test_empty_known_seats_raises(self):
        with pytest.raises(OprimError, match="known_tycoon_seats"):
            detect_hot_money_converge(
                seat_names=["方正证券成都营业部"],
                net_buy_total=8000.0,
                known_tycoon_seats=[],
            )

    def test_matched_seats_in_evidence(self):
        sig = detect_hot_money_converge(
            seat_names=["方正证券成都营业部", "国泰君安上海分公司", "other"],
            net_buy_total=9000.0,
            known_tycoon_seats=_KNOWN_SEATS,
        )
        assert sig is not None
        assert "方正证券成都营业部" in sig.evidence["matched_seats"]
        assert sig.evidence["hit_count"] == 2

    def test_high_severity_on_many_hits(self):
        cfg = HotMoneyConvergeConfig(min_seat_hits=2, min_net_buy=1000.0)
        sig = detect_hot_money_converge(
            seat_names=_KNOWN_SEATS,  # all 3 match, min=2, 3>=4 fails → medium
            net_buy_total=10000.0,
            known_tycoon_seats=_KNOWN_SEATS,
            config=cfg,
        )
        assert sig is not None
        # 3 hits < 2*2=4 → medium
        assert sig.severity == "medium"


# ── 4. detect_limit_board_explosion ──────────────────────────────────────────


class TestDetectLimitBoardExplosion:
    def _hit_data(self):
        # [10.0 → 11.0 = +10% limit_up] then [11.0 → 10.5 = -4.5% normal]
        close = [10.0, 11.0, 10.5]
        volumes = [1000.0, 2000.0, 6000.0]
        return close, volumes

    def test_hit_returns_signal(self):
        close, volumes = self._hit_data()
        sig = detect_limit_board_explosion(close=close, volumes=volumes)
        assert _is_signal(sig)
        assert sig.detector_name == "limit_board_explosion"

    def test_miss_when_both_limit_up(self):
        close = [10.0, 11.0, 12.1]  # +10%, +10% → limit_up, limit_up
        volumes = [1000.0, 2000.0, 6000.0]
        result = detect_limit_board_explosion(close=close, volumes=volumes)
        assert result is None

    def test_miss_when_volume_insufficient(self):
        close, _ = self._hit_data()
        volumes = [1000.0, 2000.0, 2000.0]  # vol_ratio=2000/1500=1.33 < 2.0
        result = detect_limit_board_explosion(close=close, volumes=volumes)
        assert result is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(OprimError, match="equal length"):
            detect_limit_board_explosion(close=[10.0, 11.0, 10.5], volumes=[1000.0, 2000.0])

    def test_too_short_raises(self):
        with pytest.raises(OprimError, match="≥ 3"):
            detect_limit_board_explosion(close=[10.0, 11.0], volumes=[1000.0, 2000.0])

    def test_evidence_vol_ratio_present(self):
        close, volumes = self._hit_data()
        sig = detect_limit_board_explosion(close=close, volumes=volumes)
        assert sig is not None
        assert "vol_ratio" in sig.evidence
        assert sig.evidence["prev_status"] == "limit_up"
        assert sig.evidence["curr_status"] == "normal"


# ── 5. detect_volume_spike ────────────────────────────────────────────────────


class TestDetectVolumeSpike:
    def _hit_data(self):
        close = [10.0] * 20 + [10.5]
        volumes = [1000.0] * 20 + [5000.0]
        return close, volumes

    def test_hit_returns_signal(self):
        close, volumes = self._hit_data()
        sig = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.015)
        assert _is_signal(sig)
        assert sig.detector_name == "volume_spike"

    def test_miss_when_vol_ratio_low(self):
        close, _ = self._hit_data()
        volumes = [1000.0] * 21  # vol_ratio = 1.0, below threshold
        result = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.015)
        assert result is None

    def test_miss_when_price_below_ma(self):
        close = [12.0] * 20 + [9.0]  # last price way below MA
        volumes = [1000.0] * 20 + [5000.0]
        result = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.015)
        assert result is None

    def test_miss_when_five_min_return_zero(self):
        close, volumes = self._hit_data()
        result = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.0)
        assert result is None

    def test_mismatched_lengths_raises(self):
        with pytest.raises(OprimError, match="equal length"):
            detect_volume_spike(close=[10.0] * 5, volumes=[1000.0] * 6, five_min_return=0.01)

    def test_evidence_contains_volume_ratio(self):
        close, volumes = self._hit_data()
        sig = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.015)
        assert sig is not None
        assert "volume_ratio" in sig.evidence
        assert sig.evidence["volume_ratio"] > 3.0

    def test_high_severity_on_extreme_vol(self):
        close = [10.0] * 20 + [10.5]
        volumes = [1000.0] * 20 + [10000.0]  # 10x → above 2× threshold
        sig = detect_volume_spike(close=close, volumes=volumes, five_min_return=0.02)
        assert sig is not None
        assert sig.severity == "high"


# ── 6. detect_northbound_reversal ─────────────────────────────────────────────


class TestDetectNorthboundReversal:
    def _hit_flows(self):
        return [1.0, 1.5, 2.0, 0.8, 1.2, -3.5]

    def test_hit_returns_signal(self):
        cfg = NorthboundReversalConfig(min_inflow_streak_mins=3, reversal_threshold=-2.0)
        sig = detect_northbound_reversal(flow_series=self._hit_flows(), config=cfg)
        assert _is_signal(sig)
        assert sig.detector_name == "northbound_reversal"

    def test_miss_when_last_bar_positive(self):
        result = detect_northbound_reversal(flow_series=[1.0, 1.5, 2.0, 0.5])
        assert result is None

    def test_miss_when_streak_too_short(self):
        cfg = NorthboundReversalConfig(min_inflow_streak_mins=5, reversal_threshold=-2.0)
        flows = [-0.5, 1.0, 1.5, -3.0]  # only 2 inflow bars before reversal
        result = detect_northbound_reversal(flow_series=flows, config=cfg)
        assert result is None

    def test_too_few_elements_raises(self):
        with pytest.raises(OprimError, match="≥ 2"):
            detect_northbound_reversal(flow_series=[-3.0])

    def test_evidence_contains_streak(self):
        cfg = NorthboundReversalConfig(min_inflow_streak_mins=3, reversal_threshold=-2.0)
        sig = detect_northbound_reversal(flow_series=self._hit_flows(), config=cfg)
        assert sig is not None
        assert "inflow_streak_mins" in sig.evidence
        assert sig.evidence["inflow_streak_mins"] == 5
        assert sig.evidence["last_flow_bn"] == pytest.approx(-3.5)

    def test_critical_on_large_reversal(self):
        cfg = NorthboundReversalConfig(min_inflow_streak_mins=3, reversal_threshold=-2.0)
        flows = [1.0, 1.5, 2.0, 1.0, -8.0]  # reversal << 2× threshold (-4.0)
        sig = detect_northbound_reversal(flow_series=flows, config=cfg)
        assert sig is not None
        assert sig.severity == "critical"


# ── 7. detect_news_shock ──────────────────────────────────────────────────────


def _mock_metrics(sentiment: float) -> list[FinancialMetric]:
    return [FinancialMetric(metric_name="net_profit", value=None, sentiment_score=sentiment)]


class TestDetectNewsShock:
    def test_hit_returns_signal(self):
        prices = [10.0, 10.3, 10.6, 10.2, 10.8]
        news = [NewsItem(content="盈利暴增")]
        with patch(
            "oprim.detect_news_shock.financial_metric_extraction", return_value=_mock_metrics(0.8)
        ):
            sig = detect_news_shock(news=news, five_min_prices=prices)
        assert _is_signal(sig)
        assert sig.detector_name == "news_shock"

    def test_miss_when_sentiment_too_low(self):
        prices = [10.0, 10.3, 10.6, 10.2, 10.8]
        news = [NewsItem(content="普通新闻")]
        with patch(
            "oprim.detect_news_shock.financial_metric_extraction", return_value=_mock_metrics(0.2)
        ):
            result = detect_news_shock(news=news, five_min_prices=prices)
        assert result is None

    def test_miss_when_volatility_low(self):
        prices = [10.0, 10.001, 10.002, 10.001, 10.002]  # near-zero vol
        news = [NewsItem(content="高情感新闻")]
        with patch(
            "oprim.detect_news_shock.financial_metric_extraction", return_value=_mock_metrics(0.9)
        ):
            cfg = NewsShockConfig(volatility_threshold=0.05)
            result = detect_news_shock(news=news, five_min_prices=prices, config=cfg)
        assert result is None

    def test_empty_news_returns_none(self):
        result = detect_news_shock(news=[], five_min_prices=[10.0, 10.5, 10.3])
        assert result is None

    def test_too_few_prices_raises(self):
        with pytest.raises(OprimError, match="≥ 2"):
            detect_news_shock(news=[NewsItem(content="x")], five_min_prices=[10.0])

    def test_evidence_contains_sentiment_and_vol(self):
        prices = [10.0, 10.4, 10.7, 10.2, 10.9]
        news = [NewsItem(content="test")]
        with patch(
            "oprim.detect_news_shock.financial_metric_extraction", return_value=_mock_metrics(0.75)
        ):
            sig = detect_news_shock(news=news, five_min_prices=prices)
        assert sig is not None
        assert "max_sentiment_score" in sig.evidence
        assert "five_min_volatility" in sig.evidence
        assert sig.evidence["max_sentiment_score"] == pytest.approx(0.75)

    def test_news_shock_stays_as_oprim(self):
        """Verify news_shock uses only financial_metric_extraction + inline vol — no other oprim."""
        import pathlib

        # Use path-based read to avoid name clash with exported function in oprim.__init__
        src = (pathlib.Path(__file__).parent.parent / "oprim" / "detect_news_shock.py").read_text()
        assert "from oprim.financial_metric_extraction import" in src
        assert "from oprim.volume_ratio import" not in src
        assert "_five_min_volatility" in src  # inline impl confirmed
