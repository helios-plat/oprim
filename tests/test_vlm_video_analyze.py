"""Tests for oprim._vlm_video_analyze."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from obase import ProviderRegistry
from oprim._vlm_video_analyze import VLMVideoAnalyzeError, vlm_video_analyze


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


@pytest.fixture()
def frames(tmp_path: Path) -> list[Path]:
    f1 = tmp_path / "frame1.png"
    f2 = tmp_path / "frame2.png"
    f1.write_bytes(b"\x89PNG\x00" * 10)
    f2.write_bytes(b"\x89PNG\x00" * 10)
    return [f1, f2]


class TestVLMVideoAnalyze:
    async def test_success(self, frames: list[Path]) -> None:
        async def _vlm(**kw: Any) -> str:
            return "A person walking"

        ProviderRegistry.register(category="vlm", name="mock_vlm", fn=_vlm)
        result = await vlm_video_analyze(
            provider="mock_vlm", frames=frames, prompt="What happens?"
        )
        assert "person" in result

    async def test_empty_frames_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VLMVideoAnalyzeError, match="frames must not be empty"):
            await vlm_video_analyze(provider="mock", frames=[], prompt="Describe")

    async def test_empty_prompt_raises(self, frames: list[Path]) -> None:
        with pytest.raises(VLMVideoAnalyzeError, match="prompt must not be empty"):
            await vlm_video_analyze(provider="mock", frames=frames, prompt="   ")

    async def test_missing_frame_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VLMVideoAnalyzeError, match="Frame not found"):
            await vlm_video_analyze(
                provider="mock",
                frames=[tmp_path / "missing.png"],
                prompt="Describe",
            )

    async def test_provider_not_found(self, frames: list[Path]) -> None:
        """Regression: unregistered VLM provider raises VLMVideoAnalyzeError."""
        with pytest.raises(VLMVideoAnalyzeError, match="VLM provider not found"):
            await vlm_video_analyze(
                provider="nonexistent_vlm", frames=frames, prompt="Describe"
            )
