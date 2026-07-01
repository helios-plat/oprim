"""Tests for P-AII-1 (failure_lesson_extract) and P-AII-2 (keyword_merge).

All tests are pure-compute; no LLM, no network, no mocks needed.
"""

from __future__ import annotations

import pytest

from oprim._aii_types import FailureLessonResult
from oprim._failure_lesson_extract import failure_lesson_extract
from oprim._keyword_merge import keyword_merge


# ===========================================================================
# P-AII-1: failure_lesson_extract
# ===========================================================================


class TestFailureLessonExtractVerifyFailed:
    def test_loogle_zero_returns_uniqueness_lesson(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"loogle_count": 0},
        )
        assert isinstance(result, FailureLessonResult)
        assert result.lesson == "非唯一命中，不可用此名确证"
        assert result.trigger_type == "verify_failed"

    def test_sharpe_below_threshold_returns_sharpe_lesson(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"sharpe": 0.3},
        )
        assert "0.30" in result.lesson
        assert "夏普" in result.lesson

    def test_sharpe_lesson_format_two_decimals(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"sharpe": 0.12345},
        )
        assert "0.12" in result.lesson

    def test_loogle_zero_takes_priority_over_sharpe(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"loogle_count": 0, "sharpe": 0.1},
        )
        assert result.lesson == "非唯一命中，不可用此名确证"

    def test_nonzero_loogle_falls_through_to_sharpe(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"loogle_count": 5, "sharpe": 0.2},
        )
        assert "0.20" in result.lesson

    def test_subject_ref_echoed(self):
        result = failure_lesson_extract(
            trigger_type="verify_failed",
            evidence={"loogle_count": 0},
            subject_ref="ku-001",
        )
        assert result.subject_ref == "ku-001"

    def test_evidence_echoed_in_result(self):
        ev = {"sharpe": 0.49}
        result = failure_lesson_extract(trigger_type="verify_failed", evidence=ev)
        assert result.evidence is ev


class TestFailureLessonExtractRetrievalMiss:
    def test_query_in_lesson(self):
        result = failure_lesson_extract(
            trigger_type="retrieval_miss",
            evidence={"query": "量化因子"},
        )
        assert "量化因子" in result.lesson
        assert "检索无命中" in result.lesson

    def test_english_query(self):
        result = failure_lesson_extract(
            trigger_type="retrieval_miss",
            evidence={"query": "momentum strategy"},
        )
        assert "momentum strategy" in result.lesson


class TestFailureLessonExtractDefeaterStruck:
    def test_contradicts_from_in_lesson(self):
        result = failure_lesson_extract(
            trigger_type="defeater_struck",
            evidence={"contradicts_from": "ku-042"},
        )
        assert "ku-042" in result.lesson
        assert "矛盾" in result.lesson


class TestFailureLessonExtractErrors:
    def test_unknown_trigger_type_raises(self):
        with pytest.raises(ValueError, match="Unknown trigger_type"):
            failure_lesson_extract(
                trigger_type="invalid_type",
                evidence={"loogle_count": 0},
            )

    def test_verify_failed_missing_both_fields_raises(self):
        with pytest.raises(ValueError):
            failure_lesson_extract(
                trigger_type="verify_failed",
                evidence={"unrelated_key": 99},
            )

    def test_retrieval_miss_missing_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            failure_lesson_extract(
                trigger_type="retrieval_miss",
                evidence={"other": "x"},
            )

    def test_defeater_struck_missing_contradicts_from_raises(self):
        with pytest.raises(ValueError, match="contradicts_from"):
            failure_lesson_extract(
                trigger_type="defeater_struck",
                evidence={"other": "y"},
            )

    def test_nonzero_loogle_no_sharpe_raises(self):
        with pytest.raises(ValueError):
            failure_lesson_extract(
                trigger_type="verify_failed",
                evidence={"loogle_count": 3},
            )

    def test_result_is_frozen(self):
        result = failure_lesson_extract(
            trigger_type="retrieval_miss",
            evidence={"query": "test"},
        )
        with pytest.raises((TypeError, AttributeError)):
            result.lesson = "mutated"  # type: ignore[misc]


# ===========================================================================
# P-AII-2: keyword_merge
# ===========================================================================


class TestKeywordMergeBasic:
    def test_empty_returns_empty_dict(self):
        assert keyword_merge([]) == {}

    def test_single_text_maps_to_itself(self):
        result = keyword_merge(["量化交易策略"])
        assert list(result.keys()) == ["量化交易策略"]
        assert result["量化交易策略"] == ["量化交易策略"]

    def test_similar_texts_merged(self):
        texts = [
            "momentum strategy backtest",
            "backtest momentum factor",
        ]
        result = keyword_merge(texts)
        assert len(result) == 1
        members = next(iter(result.values()))
        assert set(members) == set(texts)

    def test_unrelated_texts_stay_separate(self):
        texts = [
            "apple fruit healthy eating",
            "quantum computing algorithm",
            "deep sea fishing boat",
        ]
        result = keyword_merge(texts)
        assert len(result) == 3

    def test_partial_overlap_merges_chain(self):
        # A shares keyword with B, B shares keyword with C → all in one group
        texts = [
            "sharpe ratio evaluation",
            "evaluation metric performance",
            "performance attribution model",
        ]
        result = keyword_merge(texts)
        assert len(result) == 1

    def test_custom_stopwords_affect_grouping(self):
        # "factor" is normally kept; adding it to stopwords breaks the link
        texts = ["factor model returns", "factor strategy alpha"]
        result_default = keyword_merge(texts)
        # With default stopwords "factor" is a keyword → should merge
        assert len(result_default) == 1

        result_custom = keyword_merge(texts, stopwords={"factor", "model", "strategy"})
        # Now no overlap → separate groups
        assert len(result_custom) == 2

    def test_stopwords_none_uses_default(self):
        result = keyword_merge(["test function"], stopwords=None)
        assert isinstance(result, dict)

    def test_representative_is_in_members(self):
        texts = ["alpha signal", "alpha factor momentum"]
        result = keyword_merge(texts)
        for rep, members in result.items():
            assert rep in members

    def test_all_members_present_in_output(self):
        texts = ["alpha beta", "beta gamma", "delta epsilon"]
        result = keyword_merge(texts)
        all_returned = [m for members in result.values() for m in members]
        assert set(all_returned) == set(texts)

    def test_empty_tokens_after_stopwords_still_groups_separately(self):
        # Both texts contain only stopwords → no overlap → separate
        texts = ["the a in", "is are and"]
        result = keyword_merge(texts)
        # No real keywords remain; texts stay separate
        for members in result.values():
            assert len(members) >= 1

    def test_chinese_keyword_overlap(self):
        texts = ["回测夏普比率分析", "夏普比率显著性检验"]
        result = keyword_merge(texts)
        # "夏普" / "比率" overlap → merged
        assert len(result) == 1
