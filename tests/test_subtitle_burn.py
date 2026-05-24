"""Tests for oprim.subtitle_burn."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oprim.subtitle_burn import SubtitleBurnError, subtitle_burn


def _mock_ffmpeg_run():
    async def _run(*, args: list[str], **kw: object) -> str:
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


class TestSubtitleBurn:
    async def test_single_language(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        srt = tmp_path / "sub.srt"
        video.write_bytes(b"\x00" * 64)
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        out = tmp_path / "burned.mp4"
        with patch("oprim.subtitle_burn.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await subtitle_burn(video_path=video, srt_paths=[srt], output_path=out)
        assert result == out

    async def test_dual_language(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        srt_zh = tmp_path / "zh.srt"
        srt_en = tmp_path / "en.srt"
        video.write_bytes(b"\x00" * 64)
        srt_zh.write_text("1\n00:00:00,000 --> 00:00:01,000\n你好\n")
        srt_en.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
        out = tmp_path / "dual.mp4"
        with patch("oprim.subtitle_burn.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await subtitle_burn(
                video_path=video, srt_paths=[srt_zh, srt_en], output_path=out
            )
        assert result == out

    async def test_srt_not_found_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00" * 64)
        with pytest.raises(SubtitleBurnError, match="SRT file not found"):
            await subtitle_burn(
                video_path=video,
                srt_paths=[tmp_path / "missing.srt"],
                output_path=tmp_path / "out.mp4",
            )

    async def test_video_not_found_raises(self, tmp_path: Path) -> None:
        srt = tmp_path / "sub.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        with pytest.raises(SubtitleBurnError, match="Video file not found"):
            await subtitle_burn(
                video_path=tmp_path / "missing.mp4",
                srt_paths=[srt],
                output_path=tmp_path / "out.mp4",
            )

    async def test_ffmpeg_failure_raises(self, tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        video = tmp_path / "video.mp4"
        srt = tmp_path / "sub.srt"
        video.write_bytes(b"\x00" * 64)
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")

        async def _fail(**kw: object) -> str:
            raise FFmpegError("subtitle error", code=1, stderr="subtitle error")

        with patch("oprim.subtitle_burn.ffmpeg_run", side_effect=_fail):
            with pytest.raises(SubtitleBurnError, match="FFmpeg subtitle burn failed"):
                await subtitle_burn(
                    video_path=video, srt_paths=[srt], output_path=tmp_path / "out.mp4"
                )

    async def test_empty_srt_list_raises(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"\x00" * 64)
        with pytest.raises(SubtitleBurnError, match="At least one SRT"):
            await subtitle_burn(video_path=video, srt_paths=[], output_path=tmp_path / "out.mp4")

    async def test_output_created(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        srt = tmp_path / "sub.srt"
        video.write_bytes(b"\x00" * 64)
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n")
        out = tmp_path / "result.mp4"
        with patch("oprim.subtitle_burn.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            await subtitle_burn(video_path=video, srt_paths=[srt], output_path=out)
        assert out.exists()
