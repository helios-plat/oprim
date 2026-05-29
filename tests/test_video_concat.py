"""Tests for oprim.video_concat."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oprim.video_concat import VideoConcatError, video_concat


def _mock_ffmpeg_run():
    async def _run(*, args: list[str], **kw: object) -> str:
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


@pytest.fixture()
def video_files(tmp_path: Path) -> list[Path]:
    files = []
    for name in ("part1.mp4", "part2.mp4", "part3.mp4"):
        f = tmp_path / name
        f.write_bytes(b"\x00" * 64)
        files.append(f)
    return files


class TestVideoConcat:
    async def test_two_videos_demuxer(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "concat.mp4"
        with patch("oprim.video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(inputs=video_files[:2], output_path=out)
        assert result == out

    async def test_multiple_videos(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "concat3.mp4"
        with patch("oprim.video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(inputs=video_files, output_path=out)
        assert result == out

    async def test_concat_filter_method(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "filter.mp4"
        with patch("oprim.video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(
                inputs=video_files[:2], output_path=out, method="concat_filter"
            )
        assert result == out

    async def test_single_video_raises(self, video_files: list[Path], tmp_path: Path) -> None:
        with pytest.raises(VideoConcatError, match="At least 2"):
            await video_concat(inputs=video_files[:1], output_path=tmp_path / "out.mp4")

    async def test_missing_input_raises(self, tmp_path: Path) -> None:
        existing = tmp_path / "a.mp4"
        existing.write_bytes(b"\x00" * 64)
        missing = tmp_path / "missing.mp4"
        with pytest.raises(VideoConcatError, match="not found"):
            await video_concat(inputs=[existing, missing], output_path=tmp_path / "out.mp4")

    async def test_ffmpeg_failure_raises(self, video_files: list[Path], tmp_path: Path) -> None:
        from obase.ffmpeg import FFmpegError

        async def _fail(**kw: object) -> str:
            raise FFmpegError("concat error", code=1, stderr="concat error")

        with patch("oprim.video_concat.ffmpeg_run", side_effect=_fail):
            with pytest.raises(VideoConcatError, match="FFmpeg concat failed"):
                await video_concat(inputs=video_files[:2], output_path=tmp_path / "out.mp4")
