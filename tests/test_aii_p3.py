"""Tests for P-AII-3: relation_extract_rule."""
from __future__ import annotations

import pytest
from oprim._relation_extract_rule import relation_extract_rule
from oprim._aii_graph_types import RelationCandidate


class TestRelationExtractRule:
    def test_theorem_reference_returns_references(self):
        result = relation_extract_rule(
            ku_text="由傅里叶定理，任意周期函数可以展开为三角级数。",
            known_entities=["傅里叶定理"],
        )
        assert any(r.relation_type == "references" for r in result)
        assert any("傅里叶" in r.target_ref for r in result)

    def test_symbol_dep_returns_prerequisite_of(self):
        result = relation_extract_rule(
            ku_text="本节讨论该算法的收敛性。",
            ku_symbolic={"deps": ["梯度下降法"]},
            known_entities=["梯度下降法"],
        )
        assert any(r.relation_type == "prerequisite_of" for r in result)
        assert any(r.confidence_signal == "symbol_dep" for r in result)

    def test_special_case_sentence(self):
        result = relation_extract_rule(
            ku_text="本定理是高斯定理的特例，当曲面退化为平面时成立。",
        )
        assert any(r.relation_type == "special_case_of" for r in result)

    def test_no_patterns_returns_empty(self):
        result = relation_extract_rule(
            ku_text="这是一段普通的描述性文字，阐述了背景知识，没有关联关系。",
        )
        assert result == []

    def test_ambiguous_entity_gets_ambiguous_signal(self):
        result = relation_extract_rule(
            ku_text="由牛顿定理可以推导出以下结果。",
            known_entities=["牛顿第一定理", "牛顿第二定理"],
        )
        assert any(r.confidence_signal == "ambiguous" for r in result)

    def test_empty_text_returns_empty(self):
        assert relation_extract_rule(ku_text="   ") == []
        assert relation_extract_rule(ku_text="") == []

    def test_contradiction_detected(self):
        result = relation_extract_rule(
            ku_text="该结论与相对论矛盾，无法同时成立。",
        )
        assert any(r.relation_type == "contradicts" for r in result)

    def test_result_has_required_fields(self):
        result = relation_extract_rule(
            ku_text="由高斯引理得出结论。",
            known_entities=["高斯引理"],
        )
        assert result
        r = result[0]
        assert r.relation_type
        assert r.target_ref
        assert r.evidence
        assert r.confidence_signal

    def test_symbol_dep_skipped_without_known_entities(self):
        result = relation_extract_rule(
            ku_text="计算该方法。",
            ku_symbolic={"deps": ["SomeEntity"]},
            known_entities=None,
        )
        assert not any(r.relation_type == "prerequisite_of" for r in result)

    def test_degenerate_special_case(self):
        result = relation_extract_rule(
            ku_text="当参数趋近零时退化为欧氏距离。",
        )
        assert any(r.relation_type == "special_case_of" for r in result)
        assert any("欧氏距离" in r.target_ref for r in result)
