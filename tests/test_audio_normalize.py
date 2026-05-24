"""Tests for oprim.audio_normalize."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oprim.audio_normalize import AudioNormalizeError, audio_normalize


def _mock_ffmpeg_run():
    async def _run(*, args: list[str], **kw: object) -> str:
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


class TestAudioNormalize:
    async def test_normal_normalization(self, tmp_path: Path) -> None:
        inp = tmp_path / "raw.wav"
        inp.write_bytes(b"\x00" * 64)
        out = tmp_path / "norm.wav"
        with patch("oprim.audio_normalize.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_normalize(input_path=inp, output_path=out)
        assert result == out

    async def test_custom_target_lufs(self, tmp_path: Path) -> None:
        inp = tmp_path / "raw.wav"
        inp.write_bytes(b"\x00" * 64)
        out = tmp_path / "norm14.wav"
        with patch("oprim.audio_normalize.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_normalize(input_path=inp, output_path=out, target_lufs=-14.0)
        assert result == out

    async def test_input_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioNormalizeError, match="not found"):
            await audio_normalize(
                input_path=tmp_path / "missing.wav", output_path=tmp_path / "out.wav"
            )

    async def test_ffmpeg_failure_raises(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        inp = tmp_path / "raw.wav"
        inp.write_bytes(b"\x00" * 64)

        async def _fail(**kw: object) -> str:
            raise FFmpegError("filter error", code=1, stderr="filter error")

        with patch("oprim.audio_normalize.ffmpeg_run", side_effect=_fail):
            with pytest.raises(AudioNormalizeError, match="FFmpeg normalization failed"):
                await audio_normalize(input_path=inp, output_path=tmp_path / "out.wav")

    async def test_output_file_created(self, tmp_path: Path) -> None:
        inp = tmp_path / "raw.wav"
        inp.write_bytes(b"\x00" * 64)
        out = tmp_path / "verified.wav"
        with patch("oprim.audio_normalize.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            await audio_normalize(input_path=inp, output_path=out)
        assert out.exists()
