"""Tests for oprim.audio_video_merge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oprim.audio_video_merge import AudioVideoMergeError, audio_video_merge


def _mock_ffmpeg_run():
    async def _run(*, args: list[str], **kw: object) -> str:
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


class TestAudioVideoMerge:
    async def test_normal_merge(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"\x00" * 64)
        audio.write_bytes(b"\x00" * 64)
        out = tmp_path / "merged.mp4"
        with patch("oprim.audio_video_merge.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await audio_video_merge(
                video_path=video, audio_path=audio, output_path=out
            )
        assert result == out

    async def test_video_not_found_raises(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 64)
        with pytest.raises(AudioVideoMergeError, match="Video file not found"):
            await audio_video_merge(
                video_path=tmp_path / "missing.mp4",
                audio_path=audio,
                output_path=tmp_path / "out.mp4",
            )

    async def test_audio_not_found_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00" * 64)
        with pytest.raises(AudioVideoMergeError, match="Audio file not found"):
            await audio_video_merge(
                video_path=video,
                audio_path=tmp_path / "missing.wav",
                output_path=tmp_path / "out.mp4",
            )

    async def test_custom_codec(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"\x00" * 64)
        audio.write_bytes(b"\x00" * 64)
        out = tmp_path / "opus.mp4"

        captured_args: list[str] = []

        async def _capture(*, args: list[str], **kw: object) -> str:
            captured_args.extend(args)
            out.write_bytes(b"\x00" * 8)
            return "ok"

        with patch("oprim.audio_video_merge.ffmpeg_run", side_effect=_capture):
            await audio_video_merge(
                video_path=video, audio_path=audio, output_path=out, audio_codec="opus"
            )
        assert "opus" in captured_args

    async def test_ffmpeg_failure_raises(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"\x00" * 64)
        audio.write_bytes(b"\x00" * 64)

        async def _fail(**kw: object) -> str:
            raise FFmpegError("merge error", code=1, stderr="merge error")

        with patch("oprim.audio_video_merge.ffmpeg_run", side_effect=_fail):
            with pytest.raises(AudioVideoMergeError, match="FFmpeg merge failed"):
                await audio_video_merge(
                    video_path=video, audio_path=audio, output_path=tmp_path / "out.mp4"
                )

    async def test_output_created(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        audio = tmp_path / "audio.wav"
        video.write_bytes(b"\x00" * 64)
        audio.write_bytes(b"\x00" * 64)
        out = tmp_path / "result.mp4"
        with patch("oprim.audio_video_merge.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            await audio_video_merge(video_path=video, audio_path=audio, output_path=out)
        assert out.exists()
