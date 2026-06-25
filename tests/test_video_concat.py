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
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(inputs=video_files[:2], output_path=out)
        assert result == out

    async def test_multiple_videos(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "concat3.mp4"
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(inputs=video_files, output_path=out)
        assert result == out

    async def test_concat_filter_method(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "filter.mp4"
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
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

        with patch("oprim._video_concat.ffmpeg_run", side_effect=_fail):
            with pytest.raises(VideoConcatError, match="FFmpeg concat failed"):
                await video_concat(inputs=video_files[:2], output_path=tmp_path / "out.mp4")

    # ------------------------------------------------------------------
    # E6 new params — backward-compat + new behaviour
    # ------------------------------------------------------------------

    async def test_default_params_use_demuxer(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "default.mp4"
        captured_args: list[list[str]] = []

        async def _capture(**kw: object) -> str:
            captured_args.append(list(kw.get("args", [])))
            out.write_bytes(b"\x00" * 8)
            return "ok"

        with patch("oprim._video_concat.ffmpeg_run", side_effect=_capture):
            await video_concat(inputs=video_files[:2], output_path=out)

        # demuxer path uses -f concat, not filter_complex
        assert any("-f" in " ".join(a) and "concat" in " ".join(a) for a in captured_args)

    async def test_tail_frame_handling_forces_reencode(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "tail.mp4"
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(
                inputs=video_files[:2], output_path=out, tail_frame_handling=True,
            )
        assert result == out

    async def test_trim_lead_frames_forces_reencode(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "trim.mp4"
        captured: list[list[str]] = []

        async def _cap(**kw: object) -> str:
            captured.append(list(kw.get("args", [])))
            out.write_bytes(b"\x00" * 8)
            return "ok"

        with patch("oprim._video_concat.ffmpeg_run", side_effect=_cap):
            await video_concat(inputs=video_files[:2], output_path=out, trim_lead_frames=2)

        # filter_complex path — no -f concat
        flat = " ".join(" ".join(a) for a in captured)
        assert "-f concat" not in flat

    async def test_transitions_wrong_length_raises(self, video_files: list[Path], tmp_path: Path) -> None:
        with pytest.raises(VideoConcatError, match="transitions length"):
            await video_concat(
                inputs=video_files[:3],
                output_path=tmp_path / "out.mp4",
                transitions=[{"type": "hard", "duration_s": 0.0}],  # need 2, got 1
            )

    async def test_transitions_none_default_hard_cut(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "hardcut.mp4"
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(inputs=video_files[:2], output_path=out, transitions=None)
        assert result == out

    async def test_transitions_list_triggers_filter(self, video_files: list[Path], tmp_path: Path) -> None:
        out = tmp_path / "trans.mp4"
        with patch("oprim._video_concat.ffmpeg_run", side_effect=_mock_ffmpeg_run()):
            result = await video_concat(
                inputs=video_files[:2],
                output_path=out,
                transitions=[{"type": "dissolve", "duration_s": 0.5}],
            )
        assert result == out
