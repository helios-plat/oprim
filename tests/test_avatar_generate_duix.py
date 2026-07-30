"""Tests for oprim.avatar_generate duix provider (M4 — ≥5 + regression)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim.avatar_generate import AvatarGenError, avatar_generate


@pytest.fixture()
def inputs(tmp_path: Path) -> tuple[Path, Path]:
    portrait = tmp_path / "face.png"
    audio = tmp_path / "speech.wav"
    portrait.write_bytes(b"\x89PNG" + b"\x00" * 60)
    audio.write_bytes(b"RIFF" + b"\x00" * 60)
    return portrait, audio


class TestAvatarGenerateDuix:
    async def test_duix_success(self, tmp_path: Path, inputs: tuple[Path, Path]) -> None:
        """provider='duix' happy path → output file produced."""
        portrait, audio = inputs
        out = tmp_path / "avatar.mp4"

        async def _fake_submit(**kw: object) -> Path:
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 128)
            return p

        with patch(
            "oprim._providers.duix.submit_and_poll",
            new=AsyncMock(side_effect=_fake_submit),
        ):
            result = await avatar_generate(
                provider="duix",
                portrait_image=portrait,
                audio_path=audio,
                output_path=out,
            )

        assert result == out
        assert out.exists()

    async def test_duix_poll_pending_then_completed(
        self, tmp_path: Path, inputs: tuple[Path, Path]
    ) -> None:
        """Polling is handled inside submit_and_poll; provider returns completed path."""
        portrait, audio = inputs
        out = tmp_path / "polled.mp4"
        call_count = 0

        async def _fake_submit(**kw: object) -> Path:
            nonlocal call_count
            call_count += 1
            p = Path(str(kw["output_path"]))
            p.write_bytes(b"\x00" * 64)
            return p

        with patch("oprim._providers.duix.submit_and_poll", new=AsyncMock(side_effect=_fake_submit)):
            await avatar_generate(
                provider="duix",
                portrait_image=portrait,
                audio_path=audio,
                output_path=out,
            )

        assert call_count == 1

    async def test_duix_submit_failure_raises_avatar_error(
        self, tmp_path: Path, inputs: tuple[Path, Path]
    ) -> None:
        """DuixSubmitError from provider → wrapped as AvatarGenError."""
        from oprim._providers.duix import DuixSubmitError

        portrait, audio = inputs

        async def _fail(**kw: object) -> None:
            raise DuixSubmitError("HTTP 503 from duix")

        with patch("oprim._providers.duix.submit_and_poll", new=AsyncMock(side_effect=_fail)):
            with pytest.raises(AvatarGenError, match="Duix generation failed"):
                await avatar_generate(
                    provider="duix",
                    portrait_image=portrait,
                    audio_path=audio,
                    output_path=tmp_path / "out.mp4",
                )

    async def test_duix_poll_timeout_raises_avatar_error(
        self, tmp_path: Path, inputs: tuple[Path, Path]
    ) -> None:
        """DuixPollTimeoutError → wrapped as AvatarGenError."""
        from oprim._providers.duix import DuixPollTimeoutError

        portrait, audio = inputs

        async def _timeout(**kw: object) -> None:
            raise DuixPollTimeoutError("timed out after 300s")

        with patch("oprim._providers.duix.submit_and_poll", new=AsyncMock(side_effect=_timeout)):
            with pytest.raises(AvatarGenError, match="Duix generation failed"):
                await avatar_generate(
                    provider="duix",
                    portrait_image=portrait,
                    audio_path=audio,
                    output_path=tmp_path / "out.mp4",
                )

    async def test_portrait_not_found_still_validated(self, tmp_path: Path) -> None:
        """portrait_image validation runs before provider dispatch."""
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"RIFF" + b"\x00" * 60)
        with pytest.raises(AvatarGenError, match="Portrait image not found"):
            await avatar_generate(
                provider="duix",
                portrait_image=tmp_path / "missing.png",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
            )

    async def test_regression_existing_provider_wav2lip(
        self, tmp_path: Path, inputs: tuple[Path, Path]
    ) -> None:
        """Non-duix providers still route through ProviderRegistry."""
        from obase import ProviderRegistry

        ProviderRegistry.clear()
        portrait, audio = inputs
        out = tmp_path / "wav2lip.mp4"

        async def _gen(**kw: object) -> None:
            Path(str(kw["output_path"])).write_bytes(b"\x00" * 32)

        ProviderRegistry.register(category="avatar", name="wav2lip_stub", fn=_gen)
        try:
            result = await avatar_generate(
                provider="wav2lip_stub",
                portrait_image=portrait,
                audio_path=audio,
                output_path=out,
            )
            assert result == out
        finally:
            ProviderRegistry.clear()
