"""Tests for P-1 to P-4 of the Mneme speech/essay batch.

P-1: speech_to_text   (≥5)
P-2: evaluate_pronunciation  (≥5)
P-3: text_to_speech   (≥5)
P-4: rubric_score     (≥5)

All provider calls are mocked via unittest.mock — no real network.
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim._mneme_speech_types import PronunciationResult
from oprim._rubric_score import rubric_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(category: str, return_value):
    """Patch ProviderRegistry so generic(category, ...) returns a mock caller."""
    mock_reg = MagicMock()
    mock_reg.generic.return_value = return_value
    registry_patch = patch(
        f"oprim._{category.replace('_', '_')}.ProviderRegistry",
        autospec=False,
    )
    return mock_reg, registry_patch


def _valid_b64(text: str = "hello") -> str:
    return base64.b64encode(text.encode()).decode()


# ===========================================================================
# P-1: speech_to_text
# ===========================================================================

_PR_PATH = "obase.provider_registry.ProviderRegistry"


class TestSpeechToText:
    async def test_normal_recognition_returns_text(self):
        from oprim._speech_to_text import speech_to_text

        mock_caller = AsyncMock(return_value={"text": "你好世界"})
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await speech_to_text(audio_b64=_valid_b64(), language="zh")
        assert result == "你好世界"

    async def test_empty_audio_raises_value_error(self):
        from oprim._speech_to_text import speech_to_text

        with pytest.raises(ValueError, match="audio_b64"):
            await speech_to_text(audio_b64="", language="zh")

    async def test_unregistered_provider_raises_runtime_error(self):
        from oprim._speech_to_text import speech_to_text

        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.side_effect = RuntimeError(
                "asr provider 'missing' not registered"
            )
            with pytest.raises(RuntimeError):
                await speech_to_text(audio_b64=_valid_b64(), provider="missing")

    async def test_chinese_language_parameter_passed(self):
        from oprim._speech_to_text import speech_to_text

        mock_caller = AsyncMock(return_value={"text": "你好"})
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            await speech_to_text(audio_b64=_valid_b64(), language="zh")
        mock_caller.assert_awaited_once()
        kwargs = mock_caller.call_args.kwargs
        assert kwargs["language"] == "zh"

    async def test_english_language_parameter_passed(self):
        from oprim._speech_to_text import speech_to_text

        mock_caller = AsyncMock(return_value={"text": "hello world"})
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await speech_to_text(audio_b64=_valid_b64(), language="en")
        assert result == "hello world"

    async def test_string_result_returned_directly(self):
        from oprim._speech_to_text import speech_to_text

        mock_caller = AsyncMock(return_value="direct string result")
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await speech_to_text(audio_b64=_valid_b64())
        assert result == "direct string result"


# ===========================================================================
# P-2: evaluate_pronunciation
# ===========================================================================

class TestEvaluatePronunciation:
    def _mock_result(self, overall=0.85, fluency=0.80, accuracy=0.90):
        return {
            "overall_score": overall,
            "fluency_score": fluency,
            "accuracy_score": accuracy,
            "word_scores": [{"word": "hello", "score": 0.9, "issue": ""}],
        }

    async def test_normal_evaluation_returns_result(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        mock_caller = AsyncMock(return_value=self._mock_result())
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await evaluate_pronunciation(
                audio_b64=_valid_b64(), reference_text="Hello world"
            )
        assert isinstance(result, PronunciationResult)
        assert result.overall_score == pytest.approx(0.85)

    async def test_empty_reference_raises_value_error(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        with pytest.raises(ValueError, match="reference_text"):
            await evaluate_pronunciation(audio_b64=_valid_b64(), reference_text="")

    async def test_high_score_result(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        mock_caller = AsyncMock(return_value=self._mock_result(0.98, 0.97, 0.99))
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await evaluate_pronunciation(
                audio_b64=_valid_b64(), reference_text="excellent"
            )
        assert result.overall_score >= 0.9

    async def test_low_score_result(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        mock_caller = AsyncMock(return_value=self._mock_result(0.3, 0.2, 0.4))
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await evaluate_pronunciation(
                audio_b64=_valid_b64(), reference_text="difficult"
            )
        assert result.overall_score < 0.5

    async def test_word_scores_structure_complete(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        mock_caller = AsyncMock(return_value={
            "overall_score": 0.75,
            "fluency_score": 0.70,
            "accuracy_score": 0.80,
            "word_scores": [
                {"word": "this", "score": 0.8, "issue": ""},
                {"word": "is", "score": 0.9, "issue": ""},
                {"word": "hard", "score": 0.6, "issue": "consonant_cluster"},
            ],
        })
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await evaluate_pronunciation(
                audio_b64=_valid_b64(), reference_text="this is hard"
            )
        assert len(result.word_scores) == 3
        assert all("word" in ws and "score" in ws for ws in result.word_scores)

    async def test_unregistered_provider_raises(self):
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.side_effect = RuntimeError("not registered")
            with pytest.raises(RuntimeError):
                await evaluate_pronunciation(
                    audio_b64=_valid_b64(), reference_text="test", provider="missing"
                )

    async def test_pronunciation_result_passthrough(self):
        """Provider may return PronunciationResult directly."""
        from oprim._evaluate_pronunciation import evaluate_pronunciation

        pr = PronunciationResult(0.9, 0.85, 0.95, [])
        mock_caller = AsyncMock(return_value=pr)
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await evaluate_pronunciation(
                audio_b64=_valid_b64(), reference_text="ok"
            )
        assert result is pr


# ===========================================================================
# P-3: text_to_speech
# ===========================================================================

class TestTextToSpeech:
    async def test_normal_synthesis_returns_base64(self):
        from oprim._text_to_speech import text_to_speech

        audio = base64.b64encode(b"fake audio bytes").decode()
        mock_caller = AsyncMock(return_value=audio)
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await text_to_speech(text="Hello world")
        assert result == audio

    async def test_empty_text_raises_value_error(self):
        from oprim._text_to_speech import text_to_speech

        with pytest.raises(ValueError, match="text"):
            await text_to_speech(text="")

    async def test_english_voice_passed(self):
        from oprim._text_to_speech import text_to_speech

        audio = base64.b64encode(b"en audio").decode()
        mock_caller = AsyncMock(return_value=audio)
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            await text_to_speech(text="Good morning", language="en")
        kwargs = mock_caller.call_args.kwargs
        assert kwargs["language"] == "en"

    async def test_chinese_language_parameter(self):
        from oprim._text_to_speech import text_to_speech

        audio = base64.b64encode(b"zh audio").decode()
        mock_caller = AsyncMock(return_value=audio)
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await text_to_speech(text="你好", language="zh")
        assert result == audio

    async def test_base64_format_validated(self):
        from oprim._text_to_speech import text_to_speech

        mock_caller = AsyncMock(return_value="not_valid_b64!!!")
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            with pytest.raises(ValueError, match="base-64"):
                await text_to_speech(text="test")

    async def test_dict_response_audio_b64_key(self):
        from oprim._text_to_speech import text_to_speech

        audio = base64.b64encode(b"dict audio").decode()
        mock_caller = AsyncMock(return_value={"audio_b64": audio})
        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.return_value = mock_caller
            result = await text_to_speech(text="hello")
        assert result == audio

    async def test_unregistered_provider_raises(self):
        from oprim._text_to_speech import text_to_speech

        with patch(_PR_PATH) as mock_reg:
            mock_reg.get.return_value.generic.side_effect = RuntimeError("not registered")
            with pytest.raises(RuntimeError):
                await text_to_speech(text="test", provider="missing")


# ===========================================================================
# P-4: rubric_score
# ===========================================================================

_RUBRIC = {"结构": 0.25, "立意": 0.35, "语言": 0.25, "格式": 0.15}

_LONG_ESSAY = """\
近年来，随着科技的飞速发展，人工智能已经深入到我们生活的各个方面。从智能手机到自动驾驶，从医疗诊断到艺术创作，人工智能的影响无处不在。

然而，科技的发展是一把双刃剑。一方面，它为我们带来了前所未有的便利；另一方面，它也带来了一系列不可忽视的问题。

面对这些挑战，我们应当理性看待科技的发展。既不能盲目乐观，也不能因噎废食。科技本身并无善恶，关键在于人类如何使用它。

我认为，人类应当在享受科技便利的同时，保持对科技的理性反思。只有在价值观的引导下发展科技，才能让科技真正为人类服务，而不是反过来控制人类。

因此，我们既要积极拥抱科技的进步，也要时刻警醒其潜在的风险，在创新与责任之间寻求平衡。
""".strip()

_SHORT_ESSAY = "科技改变生活。"


class TestRubricScore:
    def test_normal_scoring_returns_all_dimensions(self):
        result = rubric_score(_LONG_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        assert set(result.keys()) == set(_RUBRIC.keys())

    def test_empty_rubric_raises_value_error(self):
        with pytest.raises(ValueError, match="rubric"):
            rubric_score(_LONG_ESSAY, rubric={}, grade_level="高中", essay_type="议论文")

    def test_empty_essay_returns_all_zeros(self):
        result = rubric_score("", rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        assert all(v == 0.0 for v in result.values())

    def test_all_scores_in_range_0_100(self):
        result = rubric_score(_LONG_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        for dim, score in result.items():
            assert 0.0 <= score <= 100.0, f"{dim}={score} out of range"

    def test_long_essay_scores_higher_than_short(self):
        long_result = rubric_score(_LONG_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        short_result = rubric_score(_SHORT_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        long_total = sum(long_result.values())
        short_total = sum(short_result.values())
        assert long_total > short_total

    def test_different_rubric_weights_no_effect_on_scores(self):
        """Weights are caller's responsibility; function doesn't apply them."""
        r1 = rubric_score(_LONG_ESSAY, rubric={"结构": 1.0}, grade_level="高中", essay_type="议论文")
        r2 = rubric_score(_LONG_ESSAY, rubric={"结构": 0.5}, grade_level="高中", essay_type="议论文")
        assert r1["结构"] == r2["结构"]

    def test_custom_dimension_still_returns_score(self):
        result = rubric_score(
            _LONG_ESSAY,
            rubric={"custom_dim": 1.0},
            grade_level="高中",
            essay_type="议论文",
        )
        assert "custom_dim" in result
        assert 0.0 <= result["custom_dim"] <= 100.0

    def test_short_essay_format_score_low(self):
        result = rubric_score("短", rubric={"格式": 1.0}, grade_level="高中", essay_type="议论文")
        assert result["格式"] < 70.0

    def test_deterministic_same_input_same_output(self):
        r1 = rubric_score(_LONG_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        r2 = rubric_score(_LONG_ESSAY, rubric=_RUBRIC, grade_level="高中", essay_type="议论文")
        assert r1 == r2
