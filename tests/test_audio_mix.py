"""Tests for oprim.audio_mix."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim.audio_mix import AudioMixError, audio_mix


@pytest.fixture()
def audio_files(tmp_path: Path) -> list[Path]:
    files = []
    for name in ("track1.wav", "track2.wav", "track3.wav"):
        f = tmp_path / name
        f.write_bytes(b"\x00" * 64)
        files.append(f)
    return files


def _mock_ffmpeg_run(output_path_arg_index: int = -1):
    """Return a mock that creates the expected output file."""

    async def _run(*, args: list[str], **kw: object) -> str:
        # Create the output file (last arg or expected_output)
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


class TestAudioMix:
    async def test_two_tracks(self, audio_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "mixed.wav"
        with patch("oprim.audio_mix.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_mix(inputs=audio_files[:2], output_path=out)
        assert result == out

    async def test_three_tracks(self, audio_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "mixed3.wav"
        with patch("oprim.audio_mix.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_mix(inputs=audio_files, output_path=out)
        assert result == out

    async def test_unequal_weights(self, audio_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "weighted.wav"
        with patch("oprim.audio_mix.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_mix(
                inputs=audio_files[:2], weights=[1.0, 0.3], output_path=out
            )
        assert result == out

    async def test_single_track_allowed(self, audio_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "single.wav"
        with patch("oprim.audio_mix.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_mix(inputs=audio_files[:1], output_path=out)
        assert result == out

    async def test_empty_inputs_raises(self, tmp_path: Path) -> None:
        with pytest.raises(AudioMixError, match="must not be empty"):
            await audio_mix(inputs=[], output_path=tmp_path / "out.wav")

    async def test_missing_input_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.wav"
        with pytest.raises(AudioMixError, match="not found"):
            await audio_mix(inputs=[missing], output_path=tmp_path / "out.wav")

    async def test_ffmpeg_failure_raises(self, audio_files: list[Path], tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        async def _fail(**kw: object) -> str:
            raise FFmpegError("codec error", code=1, stderr="codec error")

        out = tmp_path / "fail.wav"
        with patch("oprim.audio_mix.ffmpeg_run", side_effect=_fail):
            with pytest.raises(AudioMixError, match="FFmpeg mixing failed"):
                await audio_mix(inputs=audio_files[:2], output_path=out)

    async def test_weights_length_mismatch_raises(
        self, audio_files: list[Path], tmp_path: Path
    ) -> None:
        with pytest.raises(AudioMixError, match="weights length"):
            await audio_mix(
                inputs=audio_files[:2], weights=[1.0], output_path=tmp_path / "out.wav"
            )
