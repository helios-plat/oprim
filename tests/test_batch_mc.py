"""Tests for Mneme M-C batch: 7 education LLM elements.

All LLM calls are mocked.
Mandatory tests:
- test_kernel_takes_priority_over_llm
- test_variant_answer_empty

Version: oprim v3.5.0
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from oprim.ocr_paper import ocr_paper, OCRPaperInput, OCRPaperResult
from oprim.grade_question import grade_question, GradeQuestionInput, _compare_answer
from oprim.profiler_analyze import profiler_analyze, ProfilerInput, ProfilerResult
from oprim.socratic_turn import socratic_turn, SocraticTurnInput
from oprim.find_common_breakpoint import (
    find_common_breakpoint,
    WrongQuestion,
    BreakpointResult,
)
from oprim.generate_variant import generate_variant, VariantInput, VariantItem
from oprim.evaluate_diagram import evaluate_diagram, DiagramEvalInput, DiagramEvalResult
from oprim.types import SolveResult, SocraticTurnResult


# ─────────────────────────────────────────────────────────────────────────────
# Mock LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_caller(text: str):
    """Make a caller that returns text as LLM response."""
    from oprim.llm_complete import LLMResponse

    async def caller(**kwargs):
        return {
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    return caller


def _make_json_caller(data: dict):
    return _make_caller(json.dumps(data, ensure_ascii=False))


# ─────────────────────────────────────────────────────────────────────────────
# ocr_paper
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRPaper:
    def test_basic_ocr_from_url(self):
        response_data = {
            "raw_text": "题目1: 求 x^2 + 1 = 0 的解",
            "math_expressions": ["x^2 + 1 = 0"],
            "structured_questions": [{"number": "1", "body": "求 x^2 + 1 = 0 的解"}],
        }
        caller = _make_json_caller(response_data)
        inp = OCRPaperInput(image_url="http://example.com/paper.png")
        result = asyncio.run(ocr_paper(inp, caller=caller))
        assert result.success
        assert "x^2" in result.raw_text or "题目1" in result.raw_text
        assert len(result.math_expressions) >= 1

    def test_ocr_extracts_structured_questions(self):
        response_data = {
            "raw_text": "1. 求导",
            "math_expressions": [],
            "structured_questions": [{"number": "1", "body": "求导"}],
        }
        caller = _make_json_caller(response_data)
        inp = OCRPaperInput(image_url="http://example.com/img.jpg")
        result = asyncio.run(ocr_paper(inp, caller=caller))
        assert result.success
        assert len(result.structured_questions) == 1

    def test_ocr_no_image_fails(self):
        caller = _make_json_caller({"raw_text": "", "math_expressions": [], "structured_questions": []})
        inp = OCRPaperInput()  # no image_b64 or image_url
        result = asyncio.run(ocr_paper(inp, caller=caller))
        assert not result.success
        assert "image" in result.error.lower() or "url" in result.error.lower()

    def test_ocr_caller_error_returns_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("API error")
        inp = OCRPaperInput(image_url="http://example.com/img.jpg")
        result = asyncio.run(ocr_paper(inp, caller=failing_caller))
        assert not result.success
        assert result.error

    def test_ocr_from_b64(self):
        response_data = {
            "raw_text": "2+2",
            "math_expressions": ["2+2"],
            "structured_questions": [],
        }
        caller = _make_json_caller(response_data)
        inp = OCRPaperInput(image_b64="AAAA")
        result = asyncio.run(ocr_paper(inp, caller=caller))
        assert result.success

    def test_ocr_markdown_json_response(self):
        text = '```json\n{"raw_text": "hello", "math_expressions": [], "structured_questions": []}\n```'
        caller = _make_caller(text)
        inp = OCRPaperInput(image_url="http://example.com/img.jpg")
        result = asyncio.run(ocr_paper(inp, caller=caller))
        assert result.success
        assert result.raw_text == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# grade_question — MANDATORY test: kernel takes priority
# ─────────────────────────────────────────────────────────────────────────────

class TestGradeQuestion:
    def test_kernel_takes_priority_over_llm(self):
        """When solve_result is solvable, must NOT call LLM."""
        llm_called = []

        async def llm_spy(**kwargs):
            llm_called.append(True)
            return {
                "content": [{"type": "text", "text": '{"is_correct": true, "score": 1.0, "feedback": ""}'}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }

        solve_result = SolveResult(solvable=True, answer="42", method="kernel")
        inp = GradeQuestionInput(
            question="2+2=?",
            student_answer="42",
            solve_result=solve_result,
        )
        result = asyncio.run(grade_question(inp, caller=llm_spy))
        assert not llm_called, "LLM was called despite kernel having the answer!"
        assert result.method == "kernel"
        assert result.is_correct is True

    def test_kernel_correct_answer(self):
        solve_result = SolveResult(solvable=True, answer="6", method="kernel")
        inp = GradeQuestionInput(
            question="3*2=?",
            student_answer="6",
            solve_result=solve_result,
        )
        result = asyncio.run(grade_question(inp, caller=_make_json_caller({})))
        assert result.is_correct is True
        assert result.method == "kernel"

    def test_kernel_wrong_answer(self):
        solve_result = SolveResult(solvable=True, answer="6", method="kernel")
        inp = GradeQuestionInput(
            question="3*2=?",
            student_answer="7",
            solve_result=solve_result,
        )
        result = asyncio.run(grade_question(inp, caller=_make_json_caller({})))
        assert result.is_correct is False
        assert result.method == "kernel"

    def test_expected_answer_takes_priority(self):
        inp = GradeQuestionInput(
            question="What is 5+5?",
            student_answer="10",
            expected_answer="10",
        )
        result = asyncio.run(grade_question(inp, caller=_make_json_caller({})))
        assert result.is_correct is True
        assert result.method == "kernel"

    def test_llm_fallback_when_no_kernel(self):
        caller = _make_json_caller({"is_correct": True, "score": 0.9, "feedback": "Good"})
        inp = GradeQuestionInput(
            question="Explain something",
            student_answer="Something",
        )
        result = asyncio.run(grade_question(inp, caller=caller))
        assert result.method == "llm"
        assert result.is_correct is True

    def test_llm_error_returns_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("API error")
        inp = GradeQuestionInput(question="x=?", student_answer="1")
        result = asyncio.run(grade_question(inp, caller=failing_caller))
        assert result.method == "llm"
        assert not result.is_correct
        assert result.error

    def test_compare_answer_numeric(self):
        assert _compare_answer("3.0", "3") is True
        assert _compare_answer("3.00001", "3") is True
        assert _compare_answer("4", "3") is False

    def test_compare_answer_string(self):
        assert _compare_answer("  ABC  ", "abc") is True
        assert _compare_answer("xy", "yz") is False


# ─────────────────────────────────────────────────────────────────────────────
# profiler_analyze
# ─────────────────────────────────────────────────────────────────────────────

class TestProfilerAnalyze:
    def _make_profile_caller(self):
        data = {
            "strengths": ["algebra", "geometry"],
            "weaknesses": ["calculus"],
            "recommendations": ["Practice integrals"],
            "overall_level": "intermediate",
            "mastery_summary": "Strong in algebra but needs calculus work.",
        }
        return _make_json_caller(data)

    def test_basic_analysis(self):
        caller = self._make_profile_caller()
        inp = ProfilerInput(kc_mastery={"algebra": 0.9, "calculus": 0.3})
        result = asyncio.run(profiler_analyze(inp, caller=caller))
        assert result.success
        assert "algebra" in result.strengths or len(result.strengths) > 0
        assert len(result.weaknesses) > 0

    def test_recommendations_returned(self):
        caller = self._make_profile_caller()
        inp = ProfilerInput(kc_mastery={"k1": 0.5})
        result = asyncio.run(profiler_analyze(inp, caller=caller))
        assert result.success
        assert len(result.recommendations) > 0

    def test_overall_level(self):
        caller = self._make_profile_caller()
        inp = ProfilerInput(kc_mastery={"k1": 0.7})
        result = asyncio.run(profiler_analyze(inp, caller=caller))
        assert result.overall_level in ("beginner", "intermediate", "advanced", "unknown")

    def test_error_on_llm_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("fail")
        inp = ProfilerInput(kc_mastery={"k1": 0.5})
        result = asyncio.run(profiler_analyze(inp, caller=failing_caller))
        assert not result.success
        assert result.error

    def test_with_recent_attempts(self):
        caller = self._make_profile_caller()
        attempts = [
            {"question_id": "q1", "correct": True, "kc_id": "algebra"},
            {"question_id": "q2", "correct": False, "kc_id": "calculus"},
        ]
        inp = ProfilerInput(kc_mastery={"algebra": 0.8, "calculus": 0.2}, recent_attempts=attempts)
        result = asyncio.run(profiler_analyze(inp, caller=caller))
        assert result.success


# ─────────────────────────────────────────────────────────────────────────────
# socratic_turn
# ─────────────────────────────────────────────────────────────────────────────

class TestSocraticTurn:
    def test_basic_turn(self):
        caller = _make_caller("这道题你再想想，思路是什么？")
        inp = SocraticTurnInput(
            question="求 x^2 - 4 = 0",
            correct_answer="x = ±2",
            student_last_message="我不知道怎么做",
        )
        result = asyncio.run(socratic_turn(inp, caller=caller))
        assert isinstance(result, SocraticTurnResult)
        assert result.text

    def test_step_check_triggered_on_answer_claim(self):
        caller = _make_caller("请检查你的计算")
        inp = SocraticTurnInput(
            question="2+2=?",
            correct_answer="4",
            student_last_message="我算了，答案是5",
        )
        result = asyncio.run(socratic_turn(inp, caller=caller))
        assert result.step_check_triggered is True

    def test_no_step_check_for_general_question(self):
        caller = _make_caller("很好，继续思考")
        inp = SocraticTurnInput(
            question="x+1=3",
            correct_answer="2",
            student_last_message="我不确定怎么移项",
        )
        result = asyncio.run(socratic_turn(inp, caller=caller))
        assert result.step_check_triggered is False

    def test_llm_error_returns_fallback(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("API error")
        inp = SocraticTurnInput(
            question="x=?",
            correct_answer="1",
            student_last_message="不会",
        )
        result = asyncio.run(socratic_turn(inp, caller=failing_caller))
        assert result.text == "这道题你再想想，思路是什么？"

    def test_conversation_history_passed(self):
        history_seen = []

        async def history_capturing_caller(**kwargs):
            history_seen.extend(kwargs.get("messages", []))
            return {
                "content": [{"type": "text", "text": "继续思考"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 5},
            }

        inp = SocraticTurnInput(
            question="x+1=3",
            correct_answer="2",
            student_last_message="我觉得x=1",
            conversation_history=[
                {"role": "assistant", "content": "你觉得该怎么移项？"},
            ],
        )
        result = asyncio.run(socratic_turn(inp, caller=history_capturing_caller))
        assert len(history_seen) >= 2  # history + new message


# ─────────────────────────────────────────────────────────────────────────────
# find_common_breakpoint
# ─────────────────────────────────────────────────────────────────────────────

class TestFindCommonBreakpoint:
    def test_empty_returns_immediately(self):
        llm_called = []

        async def spy_caller(**kwargs):
            llm_called.append(True)
            return {}

        result = asyncio.run(find_common_breakpoint([], caller=spy_caller))
        assert not llm_called, "LLM called on empty wrong_questions!"
        assert result.success
        assert result.breakpoints == []

    def test_returns_breakpoints(self):
        caller = _make_json_caller({
            "breakpoints": [{"kc_id": "algebra", "error_pattern": "sign error", "frequency": 3, "description": "...", "remedy": "..."}],
            "dominant_error_type": "sign_error",
            "affected_question_ids": ["q1", "q2"],
            "summary": "Students commonly make sign errors.",
        })
        wrong = [
            WrongQuestion("q1", "x+2=5", "x=7", "x=3"),
            WrongQuestion("q2", "2x=6", "x=12", "x=3"),
        ]
        result = asyncio.run(find_common_breakpoint(wrong, caller=caller))
        assert result.success
        assert len(result.breakpoints) >= 1
        assert result.dominant_error_type == "sign_error"

    def test_error_on_llm_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("fail")
        wrong = [WrongQuestion("q1", "x=?", "2", "3")]
        result = asyncio.run(find_common_breakpoint(wrong, caller=failing_caller))
        assert not result.success
        assert result.error

    def test_affected_question_ids_returned(self):
        caller = _make_json_caller({
            "breakpoints": [],
            "dominant_error_type": "arithmetic",
            "affected_question_ids": ["q1"],
            "summary": "...",
        })
        wrong = [WrongQuestion("q1", "1+1=?", "3", "2")]
        result = asyncio.run(find_common_breakpoint(wrong, caller=caller))
        assert "q1" in result.affected_question_ids


# ─────────────────────────────────────────────────────────────────────────────
# generate_variant — MANDATORY test: answer always empty
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateVariant:
    def test_variant_answer_empty(self):
        """Generated variant answer MUST always be empty."""
        caller = _make_json_caller({
            "question": "求 x^2 - 9 = 0 的解",
            "answer": "x = ±3",   # LLM tries to set an answer
            "difficulty": "medium",
            "kc_ids": ["algebra"],
        })
        inp = VariantInput(
            original_question="求 x^2 - 4 = 0 的解",
            original_answer="x = ±2",
            kc_ids=["algebra"],
        )
        result = asyncio.run(generate_variant(inp, caller=caller))
        assert result.answer == "", (
            f"generate_variant answer must be empty, got: {result.answer!r}"
        )
        assert result.kernel_verified is False

    def test_variant_kernel_verified_false(self):
        """kernel_verified MUST always be False after generation."""
        caller = _make_json_caller({
            "question": "x+1=5, solve for x",
            "answer": "x=4",
            "difficulty": "easy",
            "kc_ids": ["linear_eq"],
        })
        inp = VariantInput(
            original_question="x+1=3, solve for x",
            original_answer="x=2",
        )
        result = asyncio.run(generate_variant(inp, caller=caller))
        assert result.kernel_verified is False

    def test_variant_question_not_empty(self):
        caller = _make_json_caller({
            "question": "2x + 1 = 7 的解",
            "answer": "x=3",
            "difficulty": "easy",
            "kc_ids": [],
        })
        inp = VariantInput(
            original_question="x + 1 = 5 的解",
            original_answer="x=4",
        )
        result = asyncio.run(generate_variant(inp, caller=caller))
        assert result.success
        assert result.question != ""

    def test_llm_error_returns_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("API error")
        inp = VariantInput(original_question="x=?", original_answer="1")
        result = asyncio.run(generate_variant(inp, caller=failing_caller))
        assert not result.success
        assert result.answer == ""       # still empty on failure
        assert result.kernel_verified is False

    def test_difficulty_preserved(self):
        caller = _make_json_caller({
            "question": "x^3=8",
            "answer": "x=2",
            "difficulty": "hard",
            "kc_ids": ["cubic_eq"],
        })
        inp = VariantInput(original_question="x^2=4", original_answer="x=2")
        result = asyncio.run(generate_variant(inp, caller=caller))
        assert result.difficulty == "hard"


# ─────────────────────────────────────────────────────────────────────────────
# evaluate_diagram
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateDiagram:
    def _good_caller(self):
        return _make_json_caller({
            "is_correct": True,
            "score": 0.9,
            "missing_elements": [],
            "extra_elements": [],
            "feedback": "图形正确，坐标标注清晰。",
            "suggestions": ["可以加粗坐标轴"],
        })

    def test_correct_diagram(self):
        caller = self._good_caller()
        inp = DiagramEvalInput(
            image_url="http://example.com/graph.png",
            question="画出 y=x^2 在 [-2,2] 的图像",
            expected_elements=["parabola", "vertex", "x-axis"],
        )
        result = asyncio.run(evaluate_diagram(inp, caller=caller))
        assert result.success
        assert result.is_correct is True
        assert result.score >= 0.8

    def test_incorrect_diagram(self):
        caller = _make_json_caller({
            "is_correct": False,
            "score": 0.3,
            "missing_elements": ["vertex", "axis_labels"],
            "extra_elements": [],
            "feedback": "缺少顶点标注",
            "suggestions": ["标注顶点坐标", "标注坐标轴"],
        })
        inp = DiagramEvalInput(
            image_url="http://example.com/bad.png",
            question="画出 y=x^2 的图像",
        )
        result = asyncio.run(evaluate_diagram(inp, caller=caller))
        assert result.success
        assert result.is_correct is False
        assert len(result.missing_elements) > 0

    def test_no_image_fails(self):
        inp = DiagramEvalInput()  # no image
        caller = self._good_caller()
        result = asyncio.run(evaluate_diagram(inp, caller=caller))
        assert not result.success
        assert result.error

    def test_suggestions_returned(self):
        caller = self._good_caller()
        inp = DiagramEvalInput(image_url="http://example.com/img.png")
        result = asyncio.run(evaluate_diagram(inp, caller=caller))
        assert isinstance(result.suggestions, list)

    def test_error_on_llm_failure(self):
        async def failing_caller(**kwargs):
            raise RuntimeError("fail")
        inp = DiagramEvalInput(image_url="http://example.com/img.png")
        result = asyncio.run(evaluate_diagram(inp, caller=failing_caller))
        assert not result.success
        assert result.error

    def test_from_b64(self):
        caller = self._good_caller()
        inp = DiagramEvalInput(image_b64="AAAA", question="plot y=x")
        result = asyncio.run(evaluate_diagram(inp, caller=caller))
        assert result.success
