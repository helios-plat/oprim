"""Tests for oprim.video_recompose."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from oprim.video_recompose import (
    VideoRecomposeError,
    VideoRecomposeSetupError,
    video_recompose,
)


def _mock_ffmpeg_run():
    async def _run(*, args: list[str], **kw: object) -> str:
        out = kw.get("expected_output") or Path(args[-1])
        out.write_bytes(b"\x00" * 8)
        return "ok"

    return _run


def _mock_probe(width: int = 1920, height: int = 1080):
    """Mock ffprobe returning given dimensions."""

    async def _exec(*cmd: str, **kw: object) -> AsyncMock:
        data = {"streams": [{"width": width, "height": height}]}
        proc = AsyncMock()
        proc.communicate = AsyncMock(return_value=(json.dumps(data).encode(), b""))
        proc.returncode = 0
        return proc

    return _exec


class TestVideoRecompose:
    async def test_landscape_to_portrait(self, tmp_path: Path) -> None:
        inp = tmp_path / "landscape.mp4"
        inp.write_bytes(b"\x00" * 64)
        out = tmp_path / "portrait.mp4"
        with (
            patch("oprim.video_recompose.shutil.which", return_value="/usr/bin/ffprobe"),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_probe(1920, 1080)),
            patch("oprim.video_recompose.ffmpeg_run", side_effect=_mock_ffmpeg_run()),
        ):
            result = await video_recompose(input_path=inp, output_path=out)
        assert result == out

    async def test_square_to_portrait(self, tmp_path: Path) -> None:
        inp = tmp_path / "square.mp4"
        inp.write_bytes(b"\x00" * 64)
        out = tmp_path / "portrait.mp4"
        with (
            patch("oprim.video_recompose.shutil.which", return_value="/usr/bin/ffprobe"),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_probe(1080, 1080)),
            patch("oprim.video_recompose.ffmpeg_run", side_effect=_mock_ffmpeg_run()),
        ):
            result = await video_recompose(input_path=inp, output_path=out)
        assert result == out

    async def test_already_target_raises(self, tmp_path: Path) -> None:
        inp = tmp_path / "portrait.mp4"
        inp.write_bytes(b"\x00" * 64)
        with (
            patch("oprim.video_recompose.shutil.which", return_value="/usr/bin/ffprobe"),
            patch("asyncio.create_subprocess_exec", side_effect=_mock_probe(1080, 1920)),
        ):
            with pytest.raises(VideoRecomposeError, match="already matches"):
                await video_recompose(input_path=inp, output_path=tmp_path / "out.mp4")

    async def test_ffprobe_missing_raises(self, tmp_path: Path) -> None:
        inp = tmp_path / "video.mp4"
        inp.write_bytes(b"\x00" * 64)
        with patch("oprim.video_recompose.shutil.which", return_value=None):
            with pytest.raises(VideoRecomposeSetupError, match="ffprobe"):
                await video_recompose(input_path=inp, output_path=tmp_path / "out.mp4")

    async def test_smart_crop_not_implemented(self, tmp_path: Path) -> None:
        inp = tmp_path / "video.mp4"
        inp.write_bytes(b"\x00" * 64)
        with pytest.raises(NotImplementedError, match="smart_crop"):
            await video_recompose(
                input_path=inp, output_path=tmp_path / "out.mp4", method="smart_crop"
            )

    async def test_input_not_found_raises(self, tmp_path: Path) -> None:
        with (
            patch("oprim.video_recompose.shutil.which", return_value="/usr/bin/ffprobe"),
        ):
            with pytest.raises(VideoRecomposeError, match="not found"):
                await video_recompose(
                    input_path=tmp_path / "missing.mp4", output_path=tmp_path / "out.mp4"
                )
