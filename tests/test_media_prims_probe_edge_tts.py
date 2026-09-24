"""Tests for oprim.probe_duration / oprim.edge_tts_word_boundary media prims."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprim import edge_tts_word_boundary, probe_duration
from oprim._edge_tts_word_boundary import EdgeTtsWordBoundaryError
from oprim._probe_duration import ProbeDurationError


class TestProbeDuration:
    def test_parses_ffprobe_stdout(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        media.write_bytes(b"x")
        mock_proc = MagicMock()
        mock_proc.stdout = "1.5\n"
        with patch("subprocess.run", return_value=mock_proc) as run:
            assert probe_duration(media) == pytest.approx(1.5)
        assert run.call_args is not None
        assert "ffprobe" in run.call_args.args[0]

    def test_missing_binary_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        media.write_bytes(b"x")
        with (
            patch("subprocess.run", side_effect=OSError("no ffprobe")),
            pytest.raises(ProbeDurationError, match="ffprobe failed"),
        ):
            probe_duration(media)

    def test_unparsable_stdout_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        media.write_bytes(b"x")
        mock_proc = MagicMock()
        mock_proc.stdout = "N/A"
        with (
            patch("subprocess.run", return_value=mock_proc),
            pytest.raises(ProbeDurationError, match="unparsable"),
        ):
            probe_duration(media)

    def test_nonzero_exit_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        media.write_bytes(b"x")
        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffprobe"),
            ),
            pytest.raises(ProbeDurationError, match="ffprobe failed"),
        ):
            probe_duration(media)


class TestEdgeTtsWordBoundary:
    def _chunks(self) -> list[dict]:
        return [
            {"type": "audio", "data": b"\xff\xf3fake"},
            {
                "type": "WordBoundary",
                "text": "你好",
                "offset": 1_000_000,
                "duration": 2_000_000,
            },
        ]

    async def test_success_writes_audio_and_converts_units(self, tmp_path: Path) -> None:
        out = tmp_path / "speech.mp3"

        class _Comm:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                for c in self._chunks():  # type: ignore[attr-defined]
                    yield c

        comm = _Comm()
        comm._chunks = lambda: self._chunks()  # type: ignore[attr-defined]

        with patch("edge_tts.Communicate", return_value=comm):
            result = await edge_tts_word_boundary(
                "你好", "zh-CN-XiaoxiaoNeural", output_path=out
            )
        assert out.exists()
        assert out.stat().st_size > 0
        assert result["audio_path"] == out
        words = result["words"]
        assert len(words) == 1
        assert words[0]["text"] == "你好"
        assert words[0]["start"] == pytest.approx(0.1)
        assert words[0]["end"] == pytest.approx(0.3)

    async def test_empty_audio_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "speech.mp3"

        class _Comm:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                if False:
                    yield {}

        with (
            patch("edge_tts.Communicate", return_value=_Comm()),
            pytest.raises(EdgeTtsWordBoundaryError, match="no audio"),
        ):
            await edge_tts_word_boundary("x", "v", output_path=out)

    async def test_stream_failure_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "speech.mp3"

        class _Comm:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                raise RuntimeError("rate limited")
                yield  # pragma: no cover

        with (
            patch("edge_tts.Communicate", return_value=_Comm()),
            pytest.raises(EdgeTtsWordBoundaryError, match="stream failed"),
        ):
            await edge_tts_word_boundary("x", "v", output_path=out)

    async def test_rate_and_pitch_forwarded(self, tmp_path: Path) -> None:
        out = tmp_path / "speech.mp3"
        sink: dict = {}

        def _factory(*args, **kwargs):
            sink["args"] = args
            sink["kwargs"] = kwargs

            class _Comm:
                async def stream(self):
                    yield {"type": "audio", "data": b"ab"}

            return _Comm()

        with patch("edge_tts.Communicate", side_effect=_factory):
            await edge_tts_word_boundary(
                "x", "v", rate="-10%", pitch="+0Hz", output_path=out
            )
        assert sink["kwargs"].get("rate") == "-10%"
        assert sink["kwargs"].get("pitch") == "+0Hz"
        assert sink["kwargs"].get("boundary") == "WordBoundary"
        assert sink["kwargs"].get("voice") == "v"
