"""Tests for B8 utility/compute oprims (13 oprims, ≥5 tests each)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._exceptions import OprimError
from oprim.compute_seat_t3_return import SeatT3ReturnResult, compute_seat_t3_return
from oprim.fetch_themes_daily import ThemeEntry, ThemesFetchError, fetch_themes_daily
from oprim.theme_to_sw_industry_mapping import ThemeSWMapping, theme_to_sw_industry_mapping
from oprim.fetch_sector_returns import SectorReturn, SectorFetchError, fetch_sector_returns
from oprim.pe_ttm_lookback_safe import PETTMResult, pe_ttm_lookback_safe
from oprim.stop_loss_compliance_check import StopLossResult, stop_loss_compliance_check
from oprim.realtime_quote_redis_fetch import (
    QuoteFetchError,
    QuoteResult,
    realtime_quote_redis_fetch,
)
from oprim.stamp_tax_rate_by_date import StampTaxResult, stamp_tax_rate_by_date
from oprim.broker_export_render import BrokerExportResult, broker_export_render
from oprim.compliance_disclaimer_inject import DISCLAIMER, compliance_disclaimer_inject
from oprim.monthly_review_jinja2_render import RenderedReport, monthly_review_jinja2_render
from oprim.train_val_oos_splitter import TrainValOOSSplit, train_val_oos_splitter
from oprim.detect_volume_dryup_breakout import VolumeBreakoutResult, detect_volume_dryup_breakout


# ── 1. compute_seat_t3_return ─────────────────────────────────────────────────


class TestComputeSeatT3Return:
    def test_profit_case(self):
        r = compute_seat_t3_return(seat_name="招商证券", buy_price=10.0, t3_price=10.5)
        assert isinstance(r, SeatT3ReturnResult)
        assert r.return_pct == pytest.approx(5.0)
        assert r.is_profit is True

    def test_loss_case(self):
        r = compute_seat_t3_return(seat_name="中信证券", buy_price=10.0, t3_price=9.0)
        assert r.return_pct == pytest.approx(-10.0)
        assert r.is_profit is False

    def test_zero_return(self):
        r = compute_seat_t3_return(seat_name="华泰", buy_price=10.0, t3_price=10.0)
        assert r.return_pct == 0.0
        assert r.is_profit is False

    def test_invalid_buy_price_raises(self):
        with pytest.raises(OprimError, match="buy_price"):
            compute_seat_t3_return(seat_name="x", buy_price=0.0, t3_price=10.0)

    def test_invalid_t3_price_raises(self):
        with pytest.raises(OprimError, match="t3_price"):
            compute_seat_t3_return(seat_name="x", buy_price=10.0, t3_price=-1.0)

    def test_return_pct_precision(self):
        r = compute_seat_t3_return(seat_name="x", buy_price=3.0, t3_price=3.1)
        assert abs(r.return_pct - 3.3333) < 0.001


# ── 2. fetch_themes_daily ─────────────────────────────────────────────────────


def _theme_rows():
    return [
        {"板块名称": "人工智能", "板块代码": "BK0811", "涨跌幅": 3.5},
        {"板块名称": "新能源", "板块代码": "BK0732", "涨跌幅": -1.2},
        {"板块名称": "消费电子", "板块代码": "BK0999", "涨跌幅": 1.8},
    ]


class TestFetchThemesDaily:
    async def test_returns_theme_entries(self):
        with patch("oprim.fetch_themes_daily._akshare_fetch_themes", return_value=_theme_rows()):
            entries = await fetch_themes_daily(as_of_date=date(2024, 3, 1))
        assert all(isinstance(e, ThemeEntry) for e in entries)
        assert len(entries) == 3

    async def test_sorted_by_change_pct_desc(self):
        with patch("oprim.fetch_themes_daily._akshare_fetch_themes", return_value=_theme_rows()):
            entries = await fetch_themes_daily()
        chgs = [e.change_pct for e in entries]
        assert chgs == sorted(chgs, reverse=True)

    async def test_unsupported_source_raises(self):
        with pytest.raises(ThemesFetchError, match="source"):
            await fetch_themes_daily(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch(
            "oprim.fetch_themes_daily._akshare_fetch_themes", side_effect=ConnectionError("x")
        ):
            with pytest.raises(ThemesFetchError, match="fetch_themes_daily"):
                await fetch_themes_daily()

    async def test_empty_rows_returns_empty(self):
        with patch("oprim.fetch_themes_daily._akshare_fetch_themes", return_value=[]):
            entries = await fetch_themes_daily()
        assert entries == []

    async def test_theme_name_and_code_preserved(self):
        with patch("oprim.fetch_themes_daily._akshare_fetch_themes", return_value=_theme_rows()):
            entries = await fetch_themes_daily()
        ai = next(e for e in entries if e.theme_name == "人工智能")
        assert ai.theme_code == "BK0811"


# ── 3. theme_to_sw_industry_mapping ──────────────────────────────────────────


class TestThemeToSWIndustryMapping:
    def test_matched_entry(self):
        table = {"人工智能": "电子", "新能源汽车": "汽车"}
        res = theme_to_sw_industry_mapping(theme_names=["人工智能"], mapping_table=table)
        assert res[0].sw_industry == "电子"
        assert res[0].matched is True

    def test_unmatched_returns_none(self):
        res = theme_to_sw_industry_mapping(theme_names=["区块链"], mapping_table={})
        assert res[0].sw_industry is None
        assert res[0].matched is False

    def test_order_preserved(self):
        table = {"A": "行业A", "B": "行业B", "C": "行业C"}
        names = ["C", "A", "B"]
        res = theme_to_sw_industry_mapping(theme_names=names, mapping_table=table)
        assert [r.theme_name for r in res] == names

    def test_mixed_match(self):
        table = {"已知": "行业X"}
        res = theme_to_sw_industry_mapping(theme_names=["已知", "未知"], mapping_table=table)
        assert res[0].matched is True
        assert res[1].matched is False

    def test_empty_names_raises(self):
        with pytest.raises(OprimError):
            theme_to_sw_industry_mapping(theme_names=[], mapping_table={})

    def test_returns_theme_sw_mapping_instances(self):
        res = theme_to_sw_industry_mapping(theme_names=["X"], mapping_table={"X": "Y"})
        assert all(isinstance(r, ThemeSWMapping) for r in res)


# ── 4. fetch_sector_returns ───────────────────────────────────────────────────


def _sector_rows():
    return [
        {"板块名称": "电子", "涨跌幅": 2.3},
        {"板块名称": "食品饮料", "涨跌幅": -0.5},
        {"板块名称": "新能源", "涨跌幅": 4.1},
        {"板块名称": "医药", "涨跌幅": 1.2},
    ]


class TestFetchSectorReturns:
    async def test_returns_sector_return_instances(self):
        with patch(
            "oprim.fetch_sector_returns._akshare_fetch_sectors", return_value=_sector_rows()
        ):
            results = await fetch_sector_returns()
        assert all(isinstance(r, SectorReturn) for r in results)
        assert len(results) == 4

    async def test_sorted_desc(self):
        with patch(
            "oprim.fetch_sector_returns._akshare_fetch_sectors", return_value=_sector_rows()
        ):
            results = await fetch_sector_returns()
        chgs = [r.change_pct for r in results]
        assert chgs == sorted(chgs, reverse=True)

    async def test_top_n_limits_results(self):
        with patch(
            "oprim.fetch_sector_returns._akshare_fetch_sectors", return_value=_sector_rows()
        ):
            results = await fetch_sector_returns(top_n=2)
        assert len(results) == 2
        assert results[0].sector_name == "新能源"

    async def test_unsupported_source_raises(self):
        with pytest.raises(SectorFetchError, match="source"):
            await fetch_sector_returns(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch(
            "oprim.fetch_sector_returns._akshare_fetch_sectors", side_effect=OSError("down")
        ):
            with pytest.raises(SectorFetchError, match="fetch_sector_returns"):
                await fetch_sector_returns()

    async def test_date_in_result(self):
        with patch(
            "oprim.fetch_sector_returns._akshare_fetch_sectors", return_value=_sector_rows()
        ):
            results = await fetch_sector_returns(as_of_date=date(2024, 3, 1))
        assert all(r.date == date(2024, 3, 1) for r in results)


# ── 5. pe_ttm_lookback_safe ───────────────────────────────────────────────────


class TestPETTMLookbackSafe:
    def _four_quarters(self):
        return [
            (date(2023, 4, 30), 0.5),
            (date(2023, 8, 31), 0.6),
            (date(2023, 11, 30), 0.55),
            (date(2024, 1, 31), 0.65),
        ]

    def test_normal_four_quarters(self):
        r = pe_ttm_lookback_safe(
            price=20.0,
            eps_quarterly=self._four_quarters(),
            as_of_date=date(2024, 3, 20),
        )
        assert isinstance(r, PETTMResult)
        assert r.quarters_used == 4
        assert r.pe_ttm == pytest.approx(20.0 / 2.3, rel=1e-4)
        assert r.warning is None

    def test_lag_filters_recent_quarter(self):
        eps = [(date(2024, 3, 10), 1.0)]  # 10 days ago < lag_days=45
        with pytest.raises(OprimError, match="lag_days"):
            pe_ttm_lookback_safe(
                price=10.0,
                eps_quarterly=eps,
                as_of_date=date(2024, 3, 20),
            )

    def test_only_three_quarters_sets_warning(self):
        three = self._four_quarters()[:3]
        r = pe_ttm_lookback_safe(
            price=15.0,
            eps_quarterly=three,
            as_of_date=date(2024, 3, 20),
        )
        assert r.quarters_used == 3
        assert r.warning is not None

    def test_negative_eps_pe_is_none(self):
        eps = [
            (date(2023, 1, 1), -0.5),
            (date(2023, 4, 1), -0.3),
            (date(2023, 7, 1), -0.2),
            (date(2023, 10, 1), -0.1),
        ]
        r = pe_ttm_lookback_safe(price=10.0, eps_quarterly=eps, as_of_date=date(2024, 3, 1))
        assert r.pe_ttm is None

    def test_price_zero_raises(self):
        with pytest.raises(OprimError, match="price"):
            pe_ttm_lookback_safe(
                price=0.0, eps_quarterly=self._four_quarters(), as_of_date=date(2024, 3, 1)
            )

    def test_no_valid_quarters_raises(self):
        eps = [(date(2024, 3, 15), 1.0)]  # only 5 days ago, fails lag
        with pytest.raises(OprimError):
            pe_ttm_lookback_safe(price=10.0, eps_quarterly=eps, as_of_date=date(2024, 3, 20))


# ── 6. stop_loss_compliance_check ────────────────────────────────────────────


class TestStopLossComplianceCheck:
    def test_triggered_on_loss(self):
        r = stop_loss_compliance_check(entry_price=10.0, current_price=9.1, stop_loss_pct=8.0)
        assert r.triggered is True
        assert r.action == "stop_loss"

    def test_not_triggered_on_profit(self):
        r = stop_loss_compliance_check(entry_price=10.0, current_price=11.0, stop_loss_pct=8.0)
        assert r.triggered is False
        assert r.action == "hold"

    def test_exactly_at_threshold(self):
        r = stop_loss_compliance_check(entry_price=10.0, current_price=9.2, stop_loss_pct=8.0)
        assert r.triggered is True

    def test_invalid_entry_price(self):
        with pytest.raises(OprimError, match="entry_price"):
            stop_loss_compliance_check(entry_price=0.0, current_price=10.0, stop_loss_pct=8.0)

    def test_invalid_stop_loss_pct(self):
        with pytest.raises(OprimError, match="stop_loss_pct"):
            stop_loss_compliance_check(entry_price=10.0, current_price=9.0, stop_loss_pct=0.0)

    def test_loss_pct_computed_correctly(self):
        r = stop_loss_compliance_check(entry_price=100.0, current_price=90.0, stop_loss_pct=15.0)
        assert r.current_loss_pct == pytest.approx(10.0)


# ── 7. realtime_quote_redis_fetch ────────────────────────────────────────────


class TestRealtimeQuoteRedisFetch:
    def _make_redis(self, return_value):
        mock = AsyncMock()
        mock.get = AsyncMock(return_value=return_value)
        return mock

    async def test_redis_hit_plain_float(self):
        redis = self._make_redis("12.34")
        r = await realtime_quote_redis_fetch(symbol="600519", redis_client=redis)
        assert r.source == "redis"
        assert r.price == pytest.approx(12.34)

    async def test_redis_hit_json_object(self):
        redis = self._make_redis('{"price": 1800.5, "ts": "2024-03-01T09:30:00"}')
        r = await realtime_quote_redis_fetch(symbol="600519", redis_client=redis)
        assert r.source == "redis"
        assert r.price == pytest.approx(1800.5)
        assert r.ts is not None

    async def test_redis_miss_uses_fallback(self):
        redis = self._make_redis(None)
        fallback = AsyncMock(return_value=1750.0)
        r = await realtime_quote_redis_fetch(
            symbol="600519", redis_client=redis, fallback_eod_fn=fallback
        )
        assert r.source == "eod_fallback"
        assert r.price == pytest.approx(1750.0)

    async def test_redis_miss_no_fallback_returns_none_source(self):
        redis = self._make_redis(None)
        r = await realtime_quote_redis_fetch(symbol="600519", redis_client=redis)
        assert r.source == "none"
        assert r.price == 0.0

    async def test_redis_exception_raises_quote_fetch_error(self):
        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=ConnectionError("refused"))
        with pytest.raises(QuoteFetchError, match="Redis GET"):
            await realtime_quote_redis_fetch(symbol="600519", redis_client=redis)

    async def test_key_prefix_used(self):
        redis = self._make_redis("9.9")
        await realtime_quote_redis_fetch(symbol="600519", redis_client=redis, key_prefix="myapp:q:")
        redis.get.assert_called_once_with("myapp:q:600519")


# ── 8. stamp_tax_rate_by_date ─────────────────────────────────────────────────


class TestStampTaxRateByDate:
    def test_sell_before_cutover(self):
        r = stamp_tax_rate_by_date(trade_date=date(2023, 1, 1), side="sell")
        assert r.rate == pytest.approx(0.001)

    def test_sell_on_cutover_day(self):
        r = stamp_tax_rate_by_date(trade_date=date(2023, 8, 28), side="sell")
        assert r.rate == pytest.approx(0.0005)

    def test_sell_after_cutover(self):
        r = stamp_tax_rate_by_date(trade_date=date(2024, 1, 1), side="sell")
        assert r.rate == pytest.approx(0.0005)

    def test_buy_always_zero(self):
        for d in [date(2020, 1, 1), date(2023, 8, 28), date(2024, 3, 1)]:
            r = stamp_tax_rate_by_date(trade_date=d, side="buy")
            assert r.rate == 0.0

    def test_invalid_side_raises(self):
        with pytest.raises(OprimError, match="side"):
            stamp_tax_rate_by_date(trade_date=date(2024, 1, 1), side="both")  # type: ignore[arg-type]

    def test_returns_stamp_tax_result(self):
        r = stamp_tax_rate_by_date(trade_date=date(2024, 1, 1), side="sell")
        assert isinstance(r, StampTaxResult)
        assert r.effective_from is not None


# ── 9. broker_export_render ───────────────────────────────────────────────────


class TestBrokerExportRender:
    def _orders(self):
        return [
            {"symbol": "600519", "qty": 100, "price": 1800.0},
            {"symbol": "000858", "qty": 200, "price": 95.0},
        ]

    def test_csv_output(self):
        cfg = {"format": "csv", "columns": ["symbol", "qty", "price"]}
        r = broker_export_render(broker_name="华泰", template_config=cfg, order_data=self._orders())
        assert r.format == "csv"
        assert "600519" in r.content
        assert r.row_count == 2

    def test_tsv_output(self):
        cfg = {"format": "tsv", "columns": ["symbol", "qty"]}
        r = broker_export_render(broker_name="国泰", template_config=cfg, order_data=self._orders())
        assert "\t" in r.content

    def test_json_output(self):
        cfg = {"format": "json"}
        r = broker_export_render(broker_name="招商", template_config=cfg, order_data=self._orders())
        assert r.format == "json"
        assert "600519" in r.content

    def test_unsupported_format_raises(self):
        with pytest.raises(OprimError, match="format"):
            broker_export_render(
                broker_name="x", template_config={"format": "xlsx"}, order_data=self._orders()
            )

    def test_missing_column_raises(self):
        cfg = {"format": "csv", "columns": ["symbol", "nonexistent"]}
        with pytest.raises(OprimError, match="nonexistent"):
            broker_export_render(broker_name="x", template_config=cfg, order_data=self._orders())

    def test_empty_orders_csv(self):
        cfg = {"format": "csv", "columns": ["symbol"]}
        r = broker_export_render(broker_name="x", template_config=cfg, order_data=[])
        assert r.row_count == 0


# ── 10. compliance_disclaimer_inject ─────────────────────────────────────────


class TestComplianceDisclaimerInject:
    def test_suffix_default(self):
        result = compliance_disclaimer_inject(text="买入 600519")
        assert result.endswith(DISCLAIMER)
        assert "买入 600519" in result

    def test_prefix(self):
        result = compliance_disclaimer_inject(text="买入 600519", position="prefix")
        assert result.startswith(DISCLAIMER)

    def test_both(self):
        result = compliance_disclaimer_inject(text="正文", position="both")
        assert result.count(DISCLAIMER) == 2

    def test_invalid_position_raises(self):
        with pytest.raises(OprimError, match="position"):
            compliance_disclaimer_inject(text="x", position="middle")  # type: ignore[arg-type]

    def test_custom_disclaimer(self):
        custom = "风险自担"
        result = compliance_disclaimer_inject(text="内容", disclaimer=custom)
        assert custom in result

    def test_disclaimer_contains_required_phrase(self):
        result = compliance_disclaimer_inject(text="任意内容")
        assert "不构成投资建议" in result


# ── 11. monthly_review_jinja2_render ─────────────────────────────────────────


class TestMonthlyReviewJinja2Render:
    def test_simple_render(self, tmp_path: Path):
        tpl = tmp_path / "review.j2"
        tpl.write_text("月份: {{ month }}, 收益: {{ return_pct }}%")
        r = monthly_review_jinja2_render(
            template_name="review.j2",
            context={"month": "2024-03", "return_pct": 3.5},
            template_dir=tmp_path,
        )
        assert isinstance(r, RenderedReport)
        assert r.content == "月份: 2024-03, 收益: 3.5%"

    def test_rendered_at_is_utc(self, tmp_path: Path):
        (tmp_path / "t.j2").write_text("ok")
        r = monthly_review_jinja2_render(template_name="t.j2", context={}, template_dir=tmp_path)
        assert r.rendered_at.tzinfo is not None
        assert r.rendered_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_template_not_found_raises(self, tmp_path: Path):
        with pytest.raises(OprimError, match="not found"):
            monthly_review_jinja2_render(
                template_name="missing.j2", context={}, template_dir=tmp_path
            )

    def test_missing_template_dir_raises(self):
        with pytest.raises(OprimError, match="template_dir"):
            monthly_review_jinja2_render(
                template_name="t.j2", context={}, template_dir="/nonexistent/path/xyz"
            )

    def test_template_name_stored(self, tmp_path: Path):
        (tmp_path / "rpt.j2").write_text("内容")
        r = monthly_review_jinja2_render(template_name="rpt.j2", context={}, template_dir=tmp_path)
        assert r.template_name == "rpt.j2"

    def test_multiline_template(self, tmp_path: Path):
        tpl = tmp_path / "m.j2"
        tpl.write_text("{% for item in items %}- {{ item }}\n{% endfor %}")
        r = monthly_review_jinja2_render(
            template_name="m.j2", context={"items": ["A", "B", "C"]}, template_dir=tmp_path
        )
        assert "- A" in r.content
        assert "- C" in r.content


# ── 12. train_val_oos_splitter ────────────────────────────────────────────────


class TestTrainValOOSSplitter:
    def test_100_items_60_20_20(self):
        s = train_val_oos_splitter(data=list(range(100)))
        assert isinstance(s, TrainValOOSSplit)
        assert len(s.train) == 60
        assert len(s.val) == 20
        assert len(s.oos) == 20

    def test_split_indices_correct(self):
        s = train_val_oos_splitter(data=list(range(100)))
        assert s.split_indices == (60, 80)

    def test_no_overlap_full_coverage(self):
        data = list(range(50))
        s = train_val_oos_splitter(data=data)
        combined = s.train + s.val + s.oos
        assert combined == data

    def test_custom_ratios(self):
        s = train_val_oos_splitter(data=list(range(100)), train_ratio=0.7, val_ratio=0.15)
        assert len(s.train) == 70
        assert len(s.oos) > 0

    def test_too_few_elements_raises(self):
        with pytest.raises(OprimError):
            train_val_oos_splitter(data=[1, 2])

    def test_ratios_sum_to_one_raises(self):
        with pytest.raises(OprimError, match="train_ratio"):
            train_val_oos_splitter(data=list(range(50)), train_ratio=0.8, val_ratio=0.2)

    def test_preserves_order(self):
        data = list(range(30))
        s = train_val_oos_splitter(data=data)
        assert s.train[-1] < s.val[0]
        assert s.val[-1] < s.oos[0]


# ── 13. detect_volume_dryup_breakout ─────────────────────────────────────────


def _make_dryup_breakout_data(n_base=20, n_dryup=5, breakout_vol=3000.0):
    """Build a synthetic price/volume series with a known dryup-breakout pattern."""
    base_vol = 1000.0
    dryup_vol = 300.0
    close = [10.0] * (n_base + n_dryup) + [10.5]
    volume = [base_vol] * n_base + [dryup_vol] * n_dryup + [breakout_vol]
    return close, volume


class TestDetectVolumeDryupBreakout:
    def test_detects_known_pattern(self):
        close, volume = _make_dryup_breakout_data()
        r = detect_volume_dryup_breakout(close=close, volume=volume)
        assert r.signal is True
        assert r.breakout_idx is not None
        assert r.dryup_start_idx is not None

    def test_no_pattern_returns_false(self):
        n = 30
        close = [10.0] * n
        volume = [1000.0] * n  # uniform volume, no dryup-breakout
        r = detect_volume_dryup_breakout(close=close, volume=volume)
        assert r.signal is False

    def test_mismatched_lengths_raises(self):
        with pytest.raises(OprimError, match="equal length"):
            detect_volume_dryup_breakout(close=[1.0] * 30, volume=[1.0] * 29)

    def test_too_short_raises(self):
        with pytest.raises(OprimError, match="less than required"):
            detect_volume_dryup_breakout(close=[1.0] * 5, volume=[1.0] * 5)

    def test_breakout_idx_is_last_bar(self):
        close, volume = _make_dryup_breakout_data()
        r = detect_volume_dryup_breakout(close=close, volume=volume)
        assert r.breakout_idx == len(close) - 1

    def test_breakout_vol_stored(self):
        close, volume = _make_dryup_breakout_data(breakout_vol=5000.0)
        r = detect_volume_dryup_breakout(close=close, volume=volume)
        assert r.signal is True
        assert r.breakout_vol == pytest.approx(5000.0)
