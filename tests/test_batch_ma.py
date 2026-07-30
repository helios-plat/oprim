"""
oprim 批次 M-A 测试套件
======================
4 个 Mneme 元素，每个 ≥5 个测试。
纯算法，无 LLM mock。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from oprim import (
    compute_effortful_gain,
    compute_effortful_gain_from_arrays,
    compute_feedback,
    compute_percentile_batch,
    compute_peer_percentile,
    grade_answer,
    recognition_update,
    recognition_update_sequence,
)
from oprim.types import (
    GradeResult,
    PeerPercentileResult,
    SolveResult,
    SolveStep,
)
from oprim.compute_effortful_gain import EffortfulGainResult
from oprim.compute_feedback import FeedbackItem
from oprim.recognition_update import RecognitionUpdateResult


# ===========================================================================
# compute_peer_percentile
# ===========================================================================

class TestComputePeerPercentile:
    def test_basic_percentile(self):
        result = compute_peer_percentile(85, [60, 70, 80, 90, 100])
        assert isinstance(result, PeerPercentileResult)
        assert result.student_value == 85
        assert 50 <= result.percentile <= 70
        assert result.peer_count == 5

    def test_top_percentile(self):
        result = compute_peer_percentile(100, [60, 70, 80, 90, 100])
        assert result.percentile >= 80
        assert result.distribution_bucket == "top_10%"

    def test_bottom_percentile(self):
        result = compute_peer_percentile(10, [20, 30, 40, 50, 60])
        assert result.percentile <= 20
        assert result.distribution_bucket in ("bottom_10%", "bottom_25%")

    def test_single_peer(self):
        result = compute_peer_percentile(50, [50])
        assert result.peer_count == 1
        # With one peer equal to student, percentile is 50% (0.5 * 100)
        assert result.percentile == 50.0

    def test_empty_peers_raises(self):
        with pytest.raises(ValueError):
            compute_peer_percentile(50, [])

    def test_modified_method(self):
        result = compute_peer_percentile(85, [60, 70, 80, 90, 100], method="modified")
        assert isinstance(result, PeerPercentileResult)
        assert 0 <= result.percentile <= 100

    def test_percentile_batch(self):
        results = compute_percentile_batch(
            {"math": 90, "reading": 70},
            {"math": [60, 70, 80, 90, 100], "reading": [50, 60, 70, 80, 90]},
        )
        assert "math" in results
        assert "reading" in results
        assert results["math"].percentile > results["reading"].percentile


# ===========================================================================
# recognition_update
# ===========================================================================

class TestRecognitionUpdate:
    def test_basic_update_correct(self):
        from oprim.recognition_update import RecognitionState
        state = RecognitionState(kc_id="kc1", p_mastery=0.3)
        result = recognition_update(state, correct=True, recognised=True)
        assert isinstance(result, RecognitionUpdateResult)
        assert result.p_mastery_after > result.p_mastery_before
        assert result.was_correct is True
        assert result.recognised is True

    def test_basic_update_incorrect(self):
        from oprim.recognition_update import RecognitionState
        state = RecognitionState(kc_id="kc1", p_mastery=0.8)
        result = recognition_update(state, correct=False, recognised=False)
        assert result.p_mastery_after < result.p_mastery_before

    def test_recognition_boost(self):
        from oprim.recognition_update import RecognitionState
        state1 = RecognitionState(kc_id="kc1", p_mastery=0.5)
        state2 = RecognitionState(kc_id="kc2", p_mastery=0.5)
        r1 = recognition_update(state1, correct=True, recognised=True)
        r2 = recognition_update(state2, correct=True, recognised=False)
        # Recognition + correct should boost more
        assert r1.p_mastery_after >= r2.p_mastery_after

    def test_sequence(self):
        from oprim.recognition_update import RecognitionState
        state = RecognitionState(kc_id="kc1", p_mastery=0.2)
        interactions = [(True, True), (True, False), (False, False), (True, True)]
        results = recognition_update_sequence(state, interactions)
        assert len(results) == 4
        # Mastery should generally increase
        assert results[-1].p_mastery_after > state.p_mastery

    def test_custom_parameters(self):
        from oprim.recognition_update import RecognitionState
        state = RecognitionState(kc_id="kc1", p_mastery=0.5)
        r = recognition_update(state, correct=True, recognised=True, p_transit=0.5)
        assert r.p_mastery_after > r.p_mastery_before

    def test_mastery_clamped(self):
        from oprim.recognition_update import RecognitionState
        state = RecognitionState(kc_id="kc1", p_mastery=0.99)
        r = recognition_update(state, correct=True, recognised=True)
        assert 0.001 <= r.p_mastery_after <= 0.999


# ===========================================================================
# compute_effortful_gain
# ===========================================================================

class TestComputeEffortfulGain:
    def test_basic_gain(self):
        result = compute_effortful_gain(
            effortful_correct_before=5,
            effortful_correct_after=8,
            effortful_total=10,
            easy_correct_before=5,
            easy_correct_after=7,
            easy_total=10,
        )
        assert isinstance(result, EffortfulGainResult)
        assert result.effortful_gain > 0
        assert result.easy_gain > 0

    def test_effortful_advantage(self):
        result = compute_effortful_gain(
            effortful_correct_before=5,
            effortful_correct_after=9,
            effortful_total=10,
            easy_correct_before=5,
            easy_correct_after=6,
            easy_total=10,
        )
        assert result.net_effortful_advantage > 0

    def test_easy_advantage(self):
        result = compute_effortful_gain(
            effortful_correct_before=5,
            effortful_correct_after=6,
            effortful_total=10,
            easy_correct_before=5,
            easy_correct_after=9,
            easy_total=10,
        )
        assert result.net_effortful_advantage < 0

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            compute_effortful_gain(0, 0, 0, 0, 0, 0)

    def test_from_arrays(self):
        effortful_before = [0, 1, 0, 1, 0]
        effortful_after = [1, 1, 1, 1, 0]
        easy_before = [0, 1, 0, 1, 0]
        easy_after = [1, 1, 0, 1, 0]
        result = compute_effortful_gain_from_arrays(
            effortful_before, effortful_after, easy_before, easy_after
        )
        assert isinstance(result, EffortfulGainResult)
        assert result.effortful_gain > result.easy_gain

    def test_confidence_low_sample(self):
        # With min_sample=3 and total=4 (< min_sample*2=6), confidence should be 0.7
        result = compute_effortful_gain(1, 2, 2, 1, 2, 2)
        assert result.confidence == 0.7

    def test_effort_ratio(self):
        result = compute_effortful_gain(5, 8, 10, 5, 7, 10)
        assert result.effort_ratio == pytest.approx(0.5)


# ===========================================================================
# compute_feedback
# ===========================================================================

class TestComputeFeedback:
    def test_correct_answer(self):
        result = compute_feedback("42", expected_answer="42")
        assert isinstance(result, FeedbackItem)
        assert result.category == "correct"
        assert result.score == 1.0

    def test_incorrect_answer(self):
        result = compute_feedback("43", expected_answer="42")
        assert result.category == "incorrect"
        assert result.score == 0.0

    def test_numeric_tolerance(self):
        result = compute_feedback("3.14159", expected_answer="3.14159265")
        assert result.category == "correct"

    def test_empty_answer(self):
        result = compute_feedback("", expected_answer="42")
        assert result.category == "format_error"

    def test_with_solve_result(self):
        sr = SolveResult(
            solvable=True,
            answer="42",
            steps=[SolveStep(step_number=1, description="Add", expression="2+2", result="4")],
        )
        result = compute_feedback("42", solve_result=sr)
        assert result.category == "correct"

    def test_fraction_comparison(self):
        result = compute_feedback("1/3", expected_answer="0.3333333333")
        assert result.category == "correct"

    def test_grade_answer_kernel(self):
        result = grade_answer("42", expected_answer="42")
        assert isinstance(result, GradeResult)
        assert result.is_correct is True
        assert result.method == "kernel"

    def test_grade_answer_llm_fallback(self):
        result = grade_answer("some complex answer")
        assert result.method == "llm"
