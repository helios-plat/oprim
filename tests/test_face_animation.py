"""Tests for oprim._face_animation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from obase import ProviderRegistry
from oprim._face_animation import (
    FaceAnimationError,
    FaceAnimationProviderNotFoundError,
    face_animation,
)


@pytest.fixture(autouse=True)
def _clean() -> None:  # type: ignore[misc]
    ProviderRegistry.clear()
    yield  # type: ignore[misc]
    ProviderRegistry.clear()


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    portrait = tmp_path / "face.png"
    audio = tmp_path / "speech.wav"
    portrait.write_bytes(b"\x89PNG\x00" * 10)
    audio.write_bytes(b"RIFF\x00" * 10)
    return portrait, audio


class TestFaceAnimation:
    async def test_success(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs
        out = tmp_path / "out.mp4"

        async def _anim(**kw: Any) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 128)

        ProviderRegistry.register(category="face_animation", name="mock_wav2lip", fn=_anim)
        result = await face_animation(
            provider="mock_wav2lip",
            portrait_image=portrait,
            audio_path=audio,
            output_path=out,
        )
        assert result == out
        assert out.exists()

    async def test_portrait_not_found(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        _, audio = inputs
        with pytest.raises(FaceAnimationError, match="Portrait not found"):
            await face_animation(
                provider="mock",
                portrait_image=tmp_path / "missing.png",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
            )

    async def test_audio_not_found(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, _ = inputs
        with pytest.raises(FaceAnimationError, match="Audio not found"):
            await face_animation(
                provider="mock",
                portrait_image=portrait,
                audio_path=tmp_path / "missing.wav",
                output_path=tmp_path / "out.mp4",
            )

    async def test_provider_not_found(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        """Regression: unregistered face_animation provider raises FaceAnimationProviderNotFoundError."""
        portrait, audio = inputs
        with pytest.raises(FaceAnimationProviderNotFoundError, match="Provider not found"):
            await face_animation(
                provider="nonexistent_anim",
                portrait_image=portrait,
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
            )
