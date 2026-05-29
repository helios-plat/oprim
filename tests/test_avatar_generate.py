"""Tests for oprim.avatar_generate."""

from __future__ import annotations

from pathlib import Path

import pytest

from obase import ProviderRegistry
from oprim.avatar_generate import AvatarGenError, AvatarSetupError, avatar_generate


@pytest.fixture(autouse=True)
def _clean():
    ProviderRegistry.clear()
    yield
    ProviderRegistry.clear()


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    portrait = tmp_path / "face.png"
    audio = tmp_path / "speech.wav"
    portrait.write_bytes(b"\x89PNG" + b"\x00" * 60)
    audio.write_bytes(b"RIFF" + b"\x00" * 60)
    return portrait, audio


class TestAvatarGenerate:
    async def test_provider_success(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs

        async def _gen(**kw: object) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 128)

        ProviderRegistry.register(category="avatar", name="mock", fn=_gen)
        out = tmp_path / "avatar.mp4"
        result = await avatar_generate(
            provider="mock", portrait_image=portrait, audio_path=audio, output_path=out
        )
        assert result == out

    async def test_portrait_not_found(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 64)
        with pytest.raises(AvatarGenError, match="Portrait image not found"):
            await avatar_generate(
                provider="mock", portrait_image=tmp_path / "missing.png",
                audio_path=audio, output_path=tmp_path / "out.mp4",
            )

    async def test_audio_not_found(self, tmp_path: Path) -> None:
        portrait = tmp_path / "face.png"
        portrait.write_bytes(b"\x00" * 64)
        with pytest.raises(AvatarGenError, match="Audio file not found"):
            await avatar_generate(
                provider="mock", portrait_image=portrait,
                audio_path=tmp_path / "missing.wav", output_path=tmp_path / "out.mp4",
            )

    async def test_provider_not_found(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs
        with pytest.raises(AvatarGenError, match="Avatar provider not found"):
            await avatar_generate(
                provider="nope", portrait_image=portrait,
                audio_path=audio, output_path=tmp_path / "out.mp4",
            )

    async def test_setup_error(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs

        async def _setup_fail(**kw: object) -> None:
            raise RuntimeError("wav2lip binary not found")

        ProviderRegistry.register(category="avatar", name="bad", fn=_setup_fail)
        with pytest.raises(AvatarSetupError, match="Vendor setup error"):
            await avatar_generate(
                provider="bad", portrait_image=portrait,
                audio_path=audio, output_path=tmp_path / "out.mp4",
            )

    async def test_subprocess_timeout(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs

        async def _timeout(**kw: object) -> None:
            raise TimeoutError("process timed out")

        ProviderRegistry.register(category="avatar", name="slow", fn=_timeout)
        with pytest.raises(AvatarGenError, match="Avatar generation failed"):
            await avatar_generate(
                provider="slow", portrait_image=portrait,
                audio_path=audio, output_path=tmp_path / "out.mp4",
            )

    async def test_output_not_produced(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        portrait, audio = inputs

        async def _noop(**kw: object) -> None:
            pass

        ProviderRegistry.register(category="avatar", name="noop", fn=_noop)
        with pytest.raises(AvatarGenError, match="did not produce"):
            await avatar_generate(
                provider="noop", portrait_image=portrait,
                audio_path=audio, output_path=tmp_path / "out.mp4",
            )
