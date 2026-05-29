"""Tests for oprim.story_predict."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from oprim.story_predict import StoryPredictError, StoryPrediction, story_predict


def _make_llm_response(forward: list[dict], backward: list[dict]) -> dict[str, Any]:
    """Build a mock LLM response dict."""
    return {"content": json.dumps({"forward": forward, "backward": backward})}


def _mock_llm(response: dict[str, Any]):
    """Return a synchronous LLM callable that always returns the given response."""

    def _llm(*, messages: list[dict]) -> dict[str, Any]:
        return response

    return _llm


@pytest.fixture()
def reference_image(tmp_path: Path) -> Path:
    img = tmp_path / "frame.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return img


class TestStoryPredict:
    async def test_forward_direction(self, reference_image: Path) -> None:
        """direction='forward': forward list populated, backward is empty."""
        llm = _mock_llm(
            _make_llm_response(
                forward=[
                    {"seconds": 3, "description": "猫向前跑"},
                    {"seconds": 5, "description": "猫跳跃"},
                ],
                backward=[{"seconds": -3, "description": "should be cleared"}],
            )
        )
        result = await story_predict(reference_image=reference_image, llm=llm, direction="forward")
        assert isinstance(result, StoryPrediction)
        assert len(result.forward) == 2
        assert result.backward == []

    async def test_backward_direction(self, reference_image: Path) -> None:
        """direction='backward': backward list populated, forward is empty."""
        llm = _mock_llm(
            _make_llm_response(
                forward=[{"seconds": 3, "description": "should be cleared"}],
                backward=[
                    {"seconds": -3, "description": "before event"},
                    {"seconds": -5, "description": "earlier event"},
                ],
            )
        )
        result = await story_predict(reference_image=reference_image, llm=llm, direction="backward")
        assert isinstance(result, StoryPrediction)
        assert result.forward == []
        assert len(result.backward) == 2

    async def test_both_directions(self, reference_image: Path) -> None:
        """direction='both': both forward and backward lists populated."""
        llm = _mock_llm(
            _make_llm_response(
                forward=[{"seconds": 3, "description": "future event"}],
                backward=[{"seconds": -3, "description": "past event"}],
            )
        )
        result = await story_predict(reference_image=reference_image, llm=llm, direction="both")
        assert len(result.forward) == 1
        assert len(result.backward) == 1

    async def test_custom_prediction_points(self, reference_image: Path) -> None:
        """prediction_points=[3,5,10] passes the custom points."""
        received_messages: list[list[dict]] = []

        def _capture_llm(*, messages: list[dict]) -> dict:
            received_messages.append(messages)
            return _make_llm_response(
                forward=[
                    {"seconds": 3, "description": "3s event"},
                    {"seconds": 5, "description": "5s event"},
                    {"seconds": 10, "description": "10s event"},
                ],
                backward=[],
            )

        result = await story_predict(
            reference_image=reference_image,
            llm=_capture_llm,
            direction="forward",
            prediction_points=[3, 5, 10],
        )
        assert len(received_messages) == 1
        # Verify the points appear in the instruction
        msg_text = str(received_messages[0])
        assert "3" in msg_text and "5" in msg_text and "10" in msg_text
        assert len(result.forward) == 3

    async def test_reference_image_not_found(self, tmp_path: Path) -> None:
        """Missing reference_image raises FileNotFoundError."""

        def _llm(*, messages: list[dict]) -> dict:
            return {}

        with pytest.raises(FileNotFoundError, match="reference_image not found"):
            await story_predict(
                reference_image=tmp_path / "missing.png",
                llm=_llm,
                direction="forward",
            )

    async def test_llm_returns_non_json_raises_story_predict_error(
        self, reference_image: Path
    ) -> None:
        """LLM returning non-JSON content raises StoryPredictError."""

        def _bad_llm(*, messages: list[dict]) -> dict:
            return {"content": "This is not JSON at all!"}

        with pytest.raises(StoryPredictError, match="invalid JSON"):
            await story_predict(reference_image=reference_image, llm=_bad_llm, direction="forward")

    async def test_pydantic_validation_failure_raises_story_predict_error(
        self, reference_image: Path
    ) -> None:
        """LLM returning JSON with wrong schema raises StoryPredictError."""

        def _bad_schema_llm(*, messages: list[dict]) -> dict:
            # 'seconds' should be int, not str
            return {
                "content": json.dumps(
                    {
                        "forward": [{"seconds": "not_int", "description": "x"}],
                        "backward": [],
                    }
                )
            }

        with pytest.raises(StoryPredictError, match="Pydantic validation failed"):
            await story_predict(
                reference_image=reference_image,
                llm=_bad_schema_llm,
                direction="forward",
            )
