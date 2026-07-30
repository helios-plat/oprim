"""Tests for B7 macro data fetch oprims (8 oprims, ≥5 tests each)."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from oprim._macro_types import MacroDataPoint, MacroFetchError
from oprim.fetch_macro_m2 import fetch_macro_m2
from oprim.fetch_macro_pboc import fetch_macro_pboc
from oprim.fetch_macro_cpi_ppi_pmi import fetch_macro_cpi_ppi_pmi
from oprim.fetch_macro_lpr import fetch_macro_lpr
from oprim.fetch_macro_rrr import fetch_macro_rrr
from oprim.fetch_macro_yield_spread import fetch_macro_yield_spread
from oprim.fetch_macro_calendar import fetch_macro_calendar
from oprim.fetch_macro_policy_news import fetch_macro_policy_news


# ── shared helpers ────────────────────────────────────────────────────────────


def _m2_rows() -> list[dict]:
    return [
        {"统计时间": "2024-01-31", "M2同比增速": 8.7, "M2": 298000.0},
        {"统计时间": "2024-02-29", "M2同比增速": 8.9, "M2": 301000.0},
        {"统计时间": "2024-03-31", "M2同比增速": 8.3, "M2": 304000.0},
    ]


def _pboc_rows() -> list[dict]:
    return [
        {"日期": "2024-01-15", "类型": "逆回购", "中标利率(%)": 1.8, "操作量(亿元)": 2000},
        {"日期": "2024-01-25", "类型": "MLF", "中标利率(%)": 2.5, "操作量(亿元)": 5000},
        {"日期": "2024-02-05", "类型": "逆回购", "中标利率(%)": 1.8, "操作量(亿元)": 1500},
    ]


def _cpi_rows() -> list[dict]:
    return [
        {"日期": "2024-01-31", "全国-同比增长": 0.7},
        {"日期": "2024-02-29", "全国-同比增长": 0.9},
    ]


def _ppi_rows() -> list[dict]:
    return [
        {"日期": "2024-01-31", "同比增长": -2.5},
        {"日期": "2024-02-29", "同比增长": -2.7},
    ]


def _pmi_rows() -> list[dict]:
    return [
        {"日期": "2024-01-31", "制造业-指数": 49.2},
        {"日期": "2024-02-29", "制造业-指数": 49.1},
    ]


def _lpr_rows() -> list[dict]:
    return [
        {"日期": "2024-01-22", "1年期LPR": 3.45, "5年期以上LPR": 4.2},
        {"日期": "2024-02-20", "1年期LPR": 3.45, "5年期以上LPR": 3.95},
    ]


def _rrr_rows() -> list[dict]:
    return [
        {"日期": "2024-02-05", "大型金融机构": 10.0, "中小金融机构": 8.0},
        {"日期": "2024-09-27", "大型金融机构": 9.5, "中小金融机构": 7.5},
    ]


def _yield_rows() -> list[dict]:
    return [
        {"日期": "2024-01-02", "中国国债收益率10年": 2.56, "美国国债收益率10年": 3.92},
        {"日期": "2024-01-03", "中国国债收益率10年": 2.55, "美国国债收益率10年": 3.90},
        {"日期": "2024-01-04", "中国国债收益率10年": 2.54, "美国国债收益率10年": 3.88},
    ]


def _calendar_rows() -> list[dict]:
    return [
        {"日期": "2024-01-11", "事件": "CPI月率", "实际值": 0.1, "预期值": 0.2, "前值": -0.1},
        {"日期": "2024-01-15", "事件": "GDP季率", "实际值": 5.2, "预期值": 5.0, "前值": 4.9},
        {"日期": "2024-02-09", "事件": "PMI", "实际值": 49.2, "预期值": 49.5, "前值": 49.0},
    ]


def _news_rows() -> list[dict]:
    return [
        {"发布时间": "2024-01-05", "标题": "央行宣布降准0.5个百分点", "摘要": "货币政策宽松"},
        {"发布时间": "2024-01-10", "标题": "财政部扩大专项债规模", "摘要": "财政政策积极"},
        {"发布时间": "2024-01-15", "标题": "科技公司发布新产品", "摘要": "非政策新闻"},
        {"发布时间": "2024-02-01", "标题": "发改委推进重大项目审批", "摘要": "发改委政策"},
    ]


# ── M2 ────────────────────────────────────────────────────────────────────────


class TestFetchMacroM2:
    async def test_returns_macro_data_points(self):
        with patch("oprim.fetch_macro_m2._akshare_fetch_m2", return_value=_m2_rows()):
            pts = await fetch_macro_m2()
        assert all(isinstance(p, MacroDataPoint) for p in pts)
        assert len(pts) == 6  # 3 dates × 2 indicators

    async def test_indicator_values_correct(self):
        with patch("oprim.fetch_macro_m2._akshare_fetch_m2", return_value=_m2_rows()):
            pts = await fetch_macro_m2()
        indicators = {p.indicator for p in pts}
        assert indicators == {"m2_yoy", "m2_abs"}

    async def test_date_range_filtering(self):
        with patch("oprim.fetch_macro_m2._akshare_fetch_m2", return_value=_m2_rows()):
            pts = await fetch_macro_m2(start_date=date(2024, 2, 1), end_date=date(2024, 2, 29))
        assert all(date(2024, 2, 1) <= p.date <= date(2024, 2, 29) for p in pts)
        assert len(pts) == 2

    async def test_source_wind_raises(self):
        with pytest.raises(MacroFetchError, match="wind"):
            await fetch_macro_m2(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps_to_macro_fetch_error(self):
        with patch(
            "oprim.fetch_macro_m2._akshare_fetch_m2", side_effect=ConnectionError("timeout")
        ):
            with pytest.raises(MacroFetchError, match="fetch_macro_m2"):
                await fetch_macro_m2()

    async def test_sorted_by_date(self):
        rows = list(reversed(_m2_rows()))
        with patch("oprim.fetch_macro_m2._akshare_fetch_m2", return_value=rows):
            pts = await fetch_macro_m2()
        dates = [p.date for p in pts]
        assert dates == sorted(dates)


# ── PBOC ─────────────────────────────────────────────────────────────────────


class TestFetchMacroPboc:
    async def test_returns_macro_data_points(self):
        with patch("oprim.fetch_macro_pboc._akshare_fetch_pboc", return_value=_pboc_rows()):
            pts = await fetch_macro_pboc()
        assert all(isinstance(p, MacroDataPoint) for p in pts)
        assert len(pts) == 3

    async def test_indicator_reverse_repo(self):
        with patch("oprim.fetch_macro_pboc._akshare_fetch_pboc", return_value=_pboc_rows()):
            pts = await fetch_macro_pboc()
        repo_pts = [p for p in pts if p.indicator == "pboc_reverse_repo_rate"]
        assert len(repo_pts) == 2
        assert repo_pts[0].value == 1.8

    async def test_mlf_in_metadata_volume(self):
        with patch("oprim.fetch_macro_pboc._akshare_fetch_pboc", return_value=_pboc_rows()):
            pts = await fetch_macro_pboc()
        mlf = [p for p in pts if p.indicator == "pboc_mlf_rate"][0]
        assert mlf.metadata["volume_bn"] == 5000

    async def test_source_tushare_raises(self):
        with pytest.raises(MacroFetchError, match="tushare"):
            await fetch_macro_pboc(source="tushare")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch(
            "oprim.fetch_macro_pboc._akshare_fetch_pboc", side_effect=OSError("unreachable")
        ):
            with pytest.raises(MacroFetchError, match="fetch_macro_pboc"):
                await fetch_macro_pboc()

    async def test_date_filter_applied(self):
        with patch("oprim.fetch_macro_pboc._akshare_fetch_pboc", return_value=_pboc_rows()):
            pts = await fetch_macro_pboc(start_date=date(2024, 1, 20))
        assert all(p.date >= date(2024, 1, 20) for p in pts)


# ── CPI / PPI / PMI ──────────────────────────────────────────────────────────


class TestFetchMacroCpiPpiPmi:
    async def test_returns_all_three_indicators(self):
        with (
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_cpi", return_value=_cpi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_ppi", return_value=_ppi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_pmi", return_value=_pmi_rows()),
        ):
            pts = await fetch_macro_cpi_ppi_pmi()
        indicators = {p.indicator for p in pts}
        assert indicators == {"cpi_yoy", "ppi_yoy", "pmi_mfg"}

    async def test_cpi_value_correct(self):
        with (
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_cpi", return_value=_cpi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_ppi", return_value=_ppi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_pmi", return_value=_pmi_rows()),
        ):
            pts = await fetch_macro_cpi_ppi_pmi()
        cpi = [p for p in pts if p.indicator == "cpi_yoy"]
        assert cpi[0].value == pytest.approx(0.7)

    async def test_date_range_filtering(self):
        with (
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_cpi", return_value=_cpi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_ppi", return_value=_ppi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_pmi", return_value=_pmi_rows()),
        ):
            pts = await fetch_macro_cpi_ppi_pmi(start_date=date(2024, 2, 1))
        assert all(p.date >= date(2024, 2, 1) for p in pts)

    async def test_source_wind_raises(self):
        with pytest.raises(MacroFetchError, match="wind"):
            await fetch_macro_cpi_ppi_pmi(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with (
            patch(
                "oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_cpi", side_effect=RuntimeError("down")
            ),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_ppi", return_value=_ppi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_pmi", return_value=_pmi_rows()),
        ):
            with pytest.raises(MacroFetchError, match="fetch_macro_cpi_ppi_pmi"):
                await fetch_macro_cpi_ppi_pmi()

    async def test_ppi_negative_value(self):
        with (
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_cpi", return_value=_cpi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_ppi", return_value=_ppi_rows()),
            patch("oprim.fetch_macro_cpi_ppi_pmi._akshare_fetch_pmi", return_value=_pmi_rows()),
        ):
            pts = await fetch_macro_cpi_ppi_pmi()
        ppi = [p for p in pts if p.indicator == "ppi_yoy"]
        assert ppi[0].value < 0


# ── LPR ──────────────────────────────────────────────────────────────────────


class TestFetchMacroLpr:
    async def test_returns_1y_and_5y(self):
        with patch("oprim.fetch_macro_lpr._akshare_fetch_lpr", return_value=_lpr_rows()):
            pts = await fetch_macro_lpr()
        assert {p.indicator for p in pts} == {"lpr_1y", "lpr_5y"}

    async def test_value_correct(self):
        with patch("oprim.fetch_macro_lpr._akshare_fetch_lpr", return_value=_lpr_rows()):
            pts = await fetch_macro_lpr()
        first_1y = [p for p in pts if p.indicator == "lpr_1y"][0]
        assert first_1y.value == pytest.approx(3.45)

    async def test_date_filter(self):
        with patch("oprim.fetch_macro_lpr._akshare_fetch_lpr", return_value=_lpr_rows()):
            pts = await fetch_macro_lpr(start_date=date(2024, 2, 1))
        assert all(p.date >= date(2024, 2, 1) for p in pts)
        assert len(pts) == 2  # 1y + 5y for 2024-02-20

    async def test_source_wind_raises(self):
        with pytest.raises(MacroFetchError):
            await fetch_macro_lpr(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch("oprim.fetch_macro_lpr._akshare_fetch_lpr", side_effect=ConnectionError("x")):
            with pytest.raises(MacroFetchError, match="fetch_macro_lpr"):
                await fetch_macro_lpr()

    async def test_5y_cut_in_metadata(self):
        with patch("oprim.fetch_macro_lpr._akshare_fetch_lpr", return_value=_lpr_rows()):
            pts = await fetch_macro_lpr()
        feb_5y = [p for p in pts if p.indicator == "lpr_5y" and p.date == date(2024, 2, 20)][0]
        assert feb_5y.value == pytest.approx(3.95)


# ── RRR ──────────────────────────────────────────────────────────────────────


class TestFetchMacroRrr:
    async def test_returns_large_and_small(self):
        with patch("oprim.fetch_macro_rrr._akshare_fetch_rrr", return_value=_rrr_rows()):
            pts = await fetch_macro_rrr()
        assert {p.indicator for p in pts} == {"rrr_large", "rrr_small"}

    async def test_value_correct(self):
        with patch("oprim.fetch_macro_rrr._akshare_fetch_rrr", return_value=_rrr_rows()):
            pts = await fetch_macro_rrr()
        large = [p for p in pts if p.indicator == "rrr_large"][0]
        assert large.value == 10.0

    async def test_date_filter(self):
        with patch("oprim.fetch_macro_rrr._akshare_fetch_rrr", return_value=_rrr_rows()):
            pts = await fetch_macro_rrr(start_date=date(2024, 9, 1))
        assert all(p.date >= date(2024, 9, 1) for p in pts)

    async def test_source_tushare_raises(self):
        with pytest.raises(MacroFetchError, match="tushare"):
            await fetch_macro_rrr(source="tushare")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch("oprim.fetch_macro_rrr._akshare_fetch_rrr", side_effect=OSError("timeout")):
            with pytest.raises(MacroFetchError, match="fetch_macro_rrr"):
                await fetch_macro_rrr()

    async def test_sorted_ascending(self):
        with patch(
            "oprim.fetch_macro_rrr._akshare_fetch_rrr", return_value=list(reversed(_rrr_rows()))
        ):
            pts = await fetch_macro_rrr()
        dates = [p.date for p in pts]
        assert dates == sorted(dates)


# ── Yield Spread ──────────────────────────────────────────────────────────────


class TestFetchMacroYieldSpread:
    async def test_returns_spread_indicator(self):
        with patch(
            "oprim.fetch_macro_yield_spread._akshare_fetch_yield_spread", return_value=_yield_rows()
        ):
            pts = await fetch_macro_yield_spread()
        assert all(p.indicator == "cn_us_yield_spread_10y" for p in pts)

    async def test_spread_computed_correctly(self):
        with patch(
            "oprim.fetch_macro_yield_spread._akshare_fetch_yield_spread", return_value=_yield_rows()
        ):
            pts = await fetch_macro_yield_spread()
        # 2.56 - 3.92 = -1.36
        assert pts[0].value == pytest.approx(-1.36, abs=1e-3)

    async def test_metadata_contains_cn_us(self):
        with patch(
            "oprim.fetch_macro_yield_spread._akshare_fetch_yield_spread", return_value=_yield_rows()
        ):
            pts = await fetch_macro_yield_spread()
        assert "cn_10y" in pts[0].metadata
        assert "us_10y" in pts[0].metadata

    async def test_date_filter(self):
        with patch(
            "oprim.fetch_macro_yield_spread._akshare_fetch_yield_spread", return_value=_yield_rows()
        ):
            pts = await fetch_macro_yield_spread(
                start_date=date(2024, 1, 3), end_date=date(2024, 1, 3)
            )
        assert len(pts) == 1
        assert pts[0].date == date(2024, 1, 3)

    async def test_source_wind_raises(self):
        with pytest.raises(MacroFetchError, match="wind"):
            await fetch_macro_yield_spread(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch(
            "oprim.fetch_macro_yield_spread._akshare_fetch_yield_spread",
            side_effect=RuntimeError("refused"),
        ):
            with pytest.raises(MacroFetchError, match="fetch_macro_yield_spread"):
                await fetch_macro_yield_spread()


# ── Calendar ─────────────────────────────────────────────────────────────────


class TestFetchMacroCalendar:
    async def test_returns_data_points(self):
        with patch(
            "oprim.fetch_macro_calendar._akshare_fetch_calendar", return_value=_calendar_rows()
        ):
            pts = await fetch_macro_calendar()
        assert len(pts) == 3
        assert all(isinstance(p, MacroDataPoint) for p in pts)

    async def test_indicator_is_event_name(self):
        with patch(
            "oprim.fetch_macro_calendar._akshare_fetch_calendar", return_value=_calendar_rows()
        ):
            pts = await fetch_macro_calendar()
        assert pts[0].indicator == "CPI月率"

    async def test_forecast_in_metadata(self):
        with patch(
            "oprim.fetch_macro_calendar._akshare_fetch_calendar", return_value=_calendar_rows()
        ):
            pts = await fetch_macro_calendar()
        assert pts[0].metadata["forecast"] == pytest.approx(0.2)
        assert pts[0].metadata["prev"] == pytest.approx(-0.1)

    async def test_date_filter(self):
        with patch(
            "oprim.fetch_macro_calendar._akshare_fetch_calendar", return_value=_calendar_rows()
        ):
            pts = await fetch_macro_calendar(start_date=date(2024, 1, 15))
        assert all(p.date >= date(2024, 1, 15) for p in pts)

    async def test_source_wind_raises(self):
        with pytest.raises(MacroFetchError, match="wind"):
            await fetch_macro_calendar(source="wind")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch("oprim.fetch_macro_calendar._akshare_fetch_calendar", side_effect=OSError("x")):
            with pytest.raises(MacroFetchError, match="fetch_macro_calendar"):
                await fetch_macro_calendar()


# ── Policy News ───────────────────────────────────────────────────────────────


class TestFetchMacroPolicyNews:
    async def test_filters_policy_relevant(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news", return_value=_news_rows()
        ):
            pts = await fetch_macro_policy_news()
        titles = [p.metadata["title"] for p in pts]
        assert "科技公司发布新产品" not in titles

    async def test_indicator_is_policy_news(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news", return_value=_news_rows()
        ):
            pts = await fetch_macro_policy_news()
        assert all(p.indicator == "policy_news" for p in pts)

    async def test_value_is_zero_sentinel(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news", return_value=_news_rows()
        ):
            pts = await fetch_macro_policy_news()
        assert all(p.value == 0.0 for p in pts)

    async def test_date_filter(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news", return_value=_news_rows()
        ):
            pts = await fetch_macro_policy_news(start_date=date(2024, 2, 1))
        assert all(p.date >= date(2024, 2, 1) for p in pts)

    async def test_source_tushare_raises(self):
        with pytest.raises(MacroFetchError, match="tushare"):
            await fetch_macro_policy_news(source="tushare")  # type: ignore[arg-type]

    async def test_network_error_wraps(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news",
            side_effect=ConnectionError("403"),
        ):
            with pytest.raises(MacroFetchError, match="fetch_macro_policy_news"):
                await fetch_macro_policy_news()

    async def test_title_in_metadata(self):
        with patch(
            "oprim.fetch_macro_policy_news._akshare_fetch_policy_news", return_value=_news_rows()
        ):
            pts = await fetch_macro_policy_news()
        assert all("title" in p.metadata for p in pts)
        assert "央行宣布降准" in pts[0].metadata["title"]
