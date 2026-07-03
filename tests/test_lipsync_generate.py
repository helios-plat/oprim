"""Tests for oprim.lipsync_generate — A4 unified dispatcher."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim._lipsync_generate import (
    LipsyncGenError,
    LipsyncNotImplementedError,
    LipsyncSetupError,
    lipsync_generate,
)


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    portrait = tmp_path / "face.png"
    audio = tmp_path / "speech.wav"
    vendor_dir = tmp_path / "vendor"
    portrait.write_bytes(b"\x89PNG" + b"\x00" * 60)
    audio.write_bytes(b"RIFF" + b"\x00" * 60)
    vendor_dir.mkdir()
    (vendor_dir / "inference.py").write_text("# stub")
    return portrait, audio, vendor_dir


class TestValidation:
    async def test_missing_audio_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LipsyncGenError, match="Audio file not found"):
            await lipsync_generate(
                provider="duix",
                audio_path=tmp_path / "missing.wav",
                output_path=tmp_path / "out.mp4",
                portrait_image=tmp_path / "face.png",
            )

    async def test_unknown_provider_raises(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, _ = inputs
        with pytest.raises(LipsyncGenError, match="Unknown lip-sync provider"):
            await lipsync_generate(
                provider="does-not-exist",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=portrait,
            )

    async def test_missing_portrait_image_raises(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        _, audio, _ = inputs
        with pytest.raises(LipsyncGenError, match="portrait_image is required"):
            await lipsync_generate(
                provider="duix", audio_path=audio, output_path=tmp_path / "out.mp4"
            )

    async def test_portrait_image_not_found_raises(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        _, audio, _ = inputs
        with pytest.raises(LipsyncGenError, match="Portrait image not found"):
            await lipsync_generate(
                provider="duix",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=tmp_path / "nope.png",
            )

    async def test_missing_vendor_dir_raises_for_sadtalker(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, _ = inputs
        with pytest.raises(LipsyncGenError, match="vendor_dir is required"):
            await lipsync_generate(
                provider="sadtalker",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=portrait,
            )


class TestVideoDrivenNotImplemented:
    @pytest.mark.parametrize("provider", ["sync", "latentsync"])
    async def test_video_driven_providers_raise_not_implemented(
        self, provider: str, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        _, audio, _ = inputs
        with pytest.raises(LipsyncNotImplementedError):
            await lipsync_generate(
                provider=provider,
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                source_video=tmp_path / "src.mp4",
            )


class TestDuixDispatch:
    async def test_duix_delegates_to_avatar_generate(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, _ = inputs
        out = tmp_path / "avatar.mp4"

        async def _fake_submit(**kw: object) -> Path:
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 64)
            return p

        with patch(
            "oprim._providers.duix.submit_and_poll",
            new=AsyncMock(side_effect=_fake_submit),
        ):
            result = await lipsync_generate(
                provider="duix",
                audio_path=audio,
                output_path=out,
                portrait_image=portrait,
            )

        assert result == out
        assert out.exists()

    async def test_duix_setup_error_mapped(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, _ = inputs
        from oprim._providers.duix import DuixError

        with (
            patch(
                "oprim._providers.duix.submit_and_poll",
                new=AsyncMock(side_effect=DuixError("vendor missing")),
            ),
            pytest.raises(LipsyncGenError),
        ):
            await lipsync_generate(
                provider="duix",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=portrait,
            )


class TestSadTalkerDispatch:
    async def test_sadtalker_success(self, tmp_path: Path, inputs: tuple[Path, Path, Path]) -> None:
        portrait, audio, vendor_dir = inputs
        out = tmp_path / "out.mp4"

        async def _fake_invoke(**kw: object) -> Path:
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 64)
            return p

        with patch("oprim._providers.sadtalker.invoke", new=AsyncMock(side_effect=_fake_invoke)):
            result = await lipsync_generate(
                provider="sadtalker",
                audio_path=audio,
                output_path=out,
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )

        assert result == out
        assert out.exists()

    async def test_sadtalker_setup_error_mapped(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, vendor_dir = inputs
        from oprim._providers.sadtalker import SadTalkerSetupError

        with (
            patch(
                "oprim._providers.sadtalker.invoke",
                new=AsyncMock(side_effect=SadTalkerSetupError("vendor missing")),
            ),
            pytest.raises(LipsyncSetupError),
        ):
            await lipsync_generate(
                provider="sadtalker",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )


class TestMuseTalkDispatch:
    async def test_musetalk_success(self, tmp_path: Path, inputs: tuple[Path, Path, Path]) -> None:
        portrait, audio, vendor_dir = inputs
        out = tmp_path / "out.mp4"

        async def _fake_invoke(**kw: object) -> Path:
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 64)
            return p

        with patch("oprim._providers.musetalk.invoke", new=AsyncMock(side_effect=_fake_invoke)):
            result = await lipsync_generate(
                provider="musetalk",
                audio_path=audio,
                output_path=out,
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )

        assert result == out
        assert out.exists()


class TestLongCatAvatarDispatch:
    async def test_longcat_avatar_success(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, vendor_dir = inputs
        out = tmp_path / "out.mp4"

        async def _fake_invoke(**kw: object) -> Path:
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 64)
            return p

        with patch(
            "oprim._providers.longcat_avatar.invoke_local", new=AsyncMock(side_effect=_fake_invoke)
        ):
            result = await lipsync_generate(
                provider="longcat_avatar",
                audio_path=audio,
                output_path=out,
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )

        assert result == out
        assert out.exists()

    async def test_longcat_avatar_gen_error_mapped(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, vendor_dir = inputs
        from oprim._providers.longcat_avatar import LongCatAvatarError

        with (
            patch(
                "oprim._providers.longcat_avatar.invoke_local",
                new=AsyncMock(side_effect=LongCatAvatarError("subprocess failed")),
            ),
            pytest.raises(LipsyncGenError),
        ):
            await lipsync_generate(
                provider="longcat_avatar",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )


class TestOutputMissing:
    async def test_provider_silently_not_writing_output_raises(
        self, tmp_path: Path, inputs: tuple[Path, Path, Path]
    ) -> None:
        portrait, audio, vendor_dir = inputs
        out = tmp_path / "never-written.mp4"

        async def _fake_invoke(**kw: object) -> Path:
            return Path(str(kw["output_path"]))  # doesn't actually write the file

        with (
            patch("oprim._providers.musetalk.invoke", new=AsyncMock(side_effect=_fake_invoke)),
            pytest.raises(LipsyncGenError, match="did not produce output"),
        ):
            await lipsync_generate(
                provider="musetalk",
                audio_path=audio,
                output_path=out,
                portrait_image=portrait,
                vendor_dir=vendor_dir,
            )
