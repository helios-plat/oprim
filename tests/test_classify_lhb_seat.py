"""Tests for oprim.classify_lhb_seat (adapted from Tide's test_seat_classification.py).

Adaptation note: Tide's original tests call ``classify_seat(name)`` directly,
relying on Tide's own hardcoded ``a_share_config`` lists (~80 tycoon seats
alone). Since classify_lhb_seat takes that market-knowledge data via an
injected ``registry=`` dict instead of importing it, these tests build a
small representative registry fixture (covering all 8 classes) rather than
duplicating Tide's full proprietary seat list into oprim's test suite —
DB-bound classify_seats_in_event()/seat_alias()-as-public-API tests are
dropped since that function/that public shape weren't ported (see module
docstring: only the pure per-seat classifier core is in scope).
"""

from __future__ import annotations

from oprim.classify_lhb_seat import SEAT_CLASSES, classify_lhb_seat

_REGISTRY = {
    "foreign_institutions": ["高盛公司有限责任公司", "摩根大通证券股份有限公司"],
    "quant_lhasa_seats": ["东方财富证券股份有限公司拉萨团结路第二证券营业部"],
    "quant_lhasa_keywords": ["拉萨", "西藏"],
    "internet_brokerages": ["国信证券股份有限公司深圳泰然九路证券营业部"],
    "internet_brokerage_keywords": ["泰然九路", "益田路荣超"],
    "tycoon_seats": ["中信证券股份有限公司上海溧阳路证券营业部"],
    "tycoon_aliases": {"中信证券股份有限公司上海溧阳路证券营业部": "溧阳路(知名游资)"},
    "tycoon_alias_keywords": {"解放南路": "宁波解放南路系"},
    "quant_keywords": ["量化", "对冲", "九坤", "幻方"],
}

# Fixture: (trader_name, expected_classification) — one representative per class.
SEAT_FIXTURES: list[tuple[str, str]] = [
    ("机构专用", "institutional"),
    ("某基金机构专用席位", "institutional"),
    ("沪股通专用", "northbound"),
    ("深股通专用", "northbound"),
    ("陆股通001", "northbound"),
    ("高盛公司有限责任公司", "foreign_institution"),
    ("东方财富证券股份有限公司拉萨团结路第二证券营业部", "quant_lhasa"),
    ("国信证券股份有限公司深圳泰然九路证券营业部", "internet_brokerage"),
    ("中信证券股份有限公司上海溧阳路证券营业部", "tycoon"),
    ("量化策略营业部", "quant"),
    ("九坤投资上海", "quant"),
    ("中信证券股份有限公司某未知小城营业部", "retail"),
    ("", "retail"),
]


class TestClassifyLhbSeat:
    def test_all_fixtures_correct(self) -> None:
        failures = []
        for trader_name, expected in SEAT_FIXTURES:
            got = classify_lhb_seat(trader_name, registry=_REGISTRY)["classification"]
            if got != expected:
                failures.append(f"'{trader_name}': expected={expected}, got={got}")
        assert not failures, "\n".join(failures)

    def test_institutional_detection(self) -> None:
        result = classify_lhb_seat("机构专用")
        assert result["classification"] == "institutional"
        assert result["confidence"] == 1.0

    def test_northbound_detection(self) -> None:
        for kw in ("沪股通", "深股通", "陆股通"):
            result = classify_lhb_seat(f"{kw}专用席位")
            assert result["classification"] == "northbound", f"Failed for {kw}"
            assert result["confidence"] == 1.0

    def test_quant_detection_needs_registry(self) -> None:
        result = classify_lhb_seat("量化策略席位", registry=_REGISTRY)
        assert result["classification"] == "quant"
        assert result["confidence"] == 0.9

    def test_no_registry_only_structural_classes_reachable(self) -> None:
        """Without a registry, quant/tycoon/etc. keyword rules can't fire —
        only northbound/institutional/retail (no market-knowledge lookup)."""
        result = classify_lhb_seat("量化策略席位")
        assert result["classification"] == "retail"

    def test_tycoon_detection(self) -> None:
        result = classify_lhb_seat("中信证券股份有限公司上海溧阳路证券营业部", registry=_REGISTRY)
        assert result["classification"] == "tycoon"
        assert result["confidence"] == 1.0

    def test_retail_fallback(self) -> None:
        result = classify_lhb_seat("某某证券某某路营业部不在名单中", registry=_REGISTRY)
        assert result["classification"] == "retail"

    def test_empty_string(self) -> None:
        result = classify_lhb_seat("")
        assert result["classification"] == "retail"

    def test_result_has_required_keys(self) -> None:
        result = classify_lhb_seat("机构专用")
        assert "classification" in result
        assert "confidence" in result
        assert "alias" in result
        assert isinstance(result["confidence"], float)

    def test_registry_override_wins(self) -> None:
        registry = {**_REGISTRY, "overrides": {"某席位": {"classification": "tycoon"}}}
        result = classify_lhb_seat("某席位", registry=registry)
        assert result["classification"] == "tycoon"
        assert result["confidence"] == 1.0

    def test_known_alias_attached(self) -> None:
        name = "中信证券股份有限公司上海溧阳路证券营业部"
        result = classify_lhb_seat(name, registry=_REGISTRY)
        assert result["alias"] is not None

    def test_alias_keyword_fallback_marks_tycoon(self) -> None:
        r = classify_lhb_seat("某证券股份有限公司宁波解放南路证券营业部", registry=_REGISTRY)
        assert r["alias"] is not None
        assert r["classification"] == "tycoon"

    def test_all_eight_classes_defined(self) -> None:
        assert len(SEAT_CLASSES) == 8
        assert set(SEAT_CLASSES) == {
            "institutional",
            "northbound",
            "foreign_institution",
            "quant_lhasa",
            "internet_brokerage",
            "tycoon",
            "quant",
            "retail",
        }
