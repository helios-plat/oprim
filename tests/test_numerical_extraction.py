"""Tests for tide._platform_ops.numerical_extraction."""

import pytest

from oprim.numerical_extraction import numerical_extraction


class TestCurrencyExtraction:
    def test_yi_yuan(self):
        result = numerical_extraction("公司营收为1.2亿元")
        exts = result["extractions"]
        currency = [e for e in exts if e["unit_type"] == "currency"]
        assert len(currency) >= 1
        assert abs(currency[0]["value"] - 1.2e8) < 1.0

    def test_wan_yuan(self):
        result = numerical_extraction("净利润12,000万")
        exts = result["extractions"]
        vals = [e["value"] for e in exts if e["unit_type"] in ("currency", "scale")]
        assert any(abs(v - 1.2e8) < 1.0 for v in vals)

    def test_plain_yuan(self):
        result = numerical_extraction("成本1,200,000元")
        exts = result["extractions"]
        assert any(abs(e["value"] - 1_200_000) < 1.0 for e in exts)


class TestPercentageExtraction:
    def test_percent_sign(self):
        result = numerical_extraction("增长30%")
        exts = result["extractions"]
        pcts = [e for e in exts if e["unit_type"] == "percentage"]
        assert len(pcts) >= 1
        assert abs(pcts[0]["value"] - 0.30) < 1e-6

    def test_baifen_zhi_form(self):
        result = numerical_extraction("下降百分之30")
        exts = result["extractions"]
        pcts = [e for e in exts if e["unit_type"] == "percentage"]
        assert len(pcts) >= 1
        assert abs(pcts[0]["value"] - 0.30) < 1e-6

    def test_ge_baifendian_not_pct(self):
        """'30个百分点' is NOT a percentage rate."""
        result = numerical_extraction("增长30个百分点")
        exts = result["extractions"]
        pcts = [e for e in exts if e["unit_type"] == "percentage"]
        assert len(pcts) == 0


class TestScaleExtraction:
    def test_yi(self):
        result = numerical_extraction("规模达到5亿")
        exts = result["extractions"]
        vals = [e["value"] for e in exts if e["unit_type"] in ("currency", "scale")]
        assert any(abs(v - 5e8) < 1.0 for v in vals)


class TestRatioExtraction:
    def test_multiplier(self):
        result = numerical_extraction("市盈率12.5倍", units=["ratio"])
        exts = result["extractions"]
        ratios = [e for e in exts if e["unit_type"] == "ratio"]
        assert len(ratios) >= 1
        assert abs(ratios[0]["value"] - 12.5) < 1e-6


class TestDateExtraction:
    def test_year_month(self):
        result = numerical_extraction("2024年3月报告", units=["date"])
        exts = result["extractions"]
        dates = [e for e in exts if e["unit_type"] == "date"]
        assert len(dates) >= 1

    def test_quarter(self):
        result = numerical_extraction("Q3 季度利润", units=["date"])
        exts = result["extractions"]
        dates = [e for e in exts if e["unit_type"] == "date"]
        assert len(dates) >= 1


class TestTraditionalNumerals:
    def test_wan_numeral(self):
        result = numerical_extraction("壹仟万元")
        exts = result["extractions"]
        # Should extract 1000 (万) × 1e4 = 1e7 or 1000 as scale
        assert len(exts) >= 1


class TestReturnSpans:
    def test_span_returned(self):
        result = numerical_extraction("营收1.2亿元增长30%", return_spans=True)
        for e in result["extractions"]:
            assert "span" in e
            start, end = e["span"]
            assert 0 <= start < end


class TestScaleDeduplication:
    def test_scale_not_double_counted_as_currency(self):
        # "5亿元" should appear once (currency) not also as scale
        result = numerical_extraction("规模达到5亿元", units=["currency", "scale"])
        exts = result["extractions"]
        texts = [e["original_text"] for e in exts]
        # No exact duplicate texts
        assert len(texts) == len(set(texts))

    def test_scale_no_currency_suffix(self):
        result = numerical_extraction("市值5亿增长了", units=["scale"])
        exts = result["extractions"]
        assert any(abs(e["value"] - 5e8) < 1.0 for e in exts if e["unit_type"] == "scale")

    def test_all_units_mode(self):
        text = "营收1.2亿元，增长30%，市盈率12倍，2024年1月"
        result = numerical_extraction(text, units=["currency", "percentage", "ratio", "date"])
        types = {e["unit_type"] for e in result["extractions"]}
        assert "currency" in types or "percentage" in types

    def test_baifen_zhi_group2(self):
        result = numerical_extraction("下降百分之50", units=["percentage"])
        exts = result["extractions"]
        pcts = [e for e in exts if e["unit_type"] == "percentage"]
        assert len(pcts) >= 1
        assert abs(pcts[0]["value"] - 0.50) < 1e-6

    def test_custom_scale_mapping(self):
        custom = {"百万": 1e6, "万": 1e4}
        result = numerical_extraction("交易额5百万", units=["scale"], scale_mapping=custom)
        exts = result["extractions"]
        assert any(abs(e["value"] - 5e6) < 1.0 for e in exts)


class TestValidation:
    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            numerical_extraction("")

    def test_summary_counts(self):
        result = numerical_extraction("营收1.2亿元, 增长30%, 市盈率15倍")
        assert isinstance(result["summary"], dict)
        total = sum(result["summary"].values())
        assert total == len(result["extractions"])
