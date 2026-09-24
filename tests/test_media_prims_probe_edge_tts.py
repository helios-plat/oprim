"""Tests for oprim.probe_duration / oprim.edge_tts_word_boundary media prims."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from oprim import edge_tts_word_boundary, probe_duration
from oprim._edge_tts_word_boundary import EdgeTtsWordBoundaryError
from oprim._media_probe import MediaInfo
from oprim._probe_duration import ProbeDurationError

_MEDIA_PROBE = "oprim._probe_duration.media_probe"


class TestProbeDuration:
    def test_delegates_to_media_probe(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        with patch(
            _MEDIA_PROBE, return_value=MediaInfo(duration_seconds=1.5)
        ) as mocked:
            assert probe_duration(media) == pytest.approx(1.5)
        mocked.assert_called_once_with(path=str(media))

    def test_missing_duration_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        with (
            patch(_MEDIA_PROBE, return_value=MediaInfo(duration_seconds=None)),
            pytest.raises(ProbeDurationError, match="missing"),
        ):
            probe_duration(media)

    def test_unparsable_duration_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        with (
            patch(
                _MEDIA_PROBE,
                return_value=SimpleNamespace(duration_seconds="N/A"),
            ),
            pytest.raises(ProbeDurationError, match="unparsable"),
        ):
            probe_duration(media)

    def test_probe_failure_wrapped(self, tmp_path: Path) -> None:
        media = tmp_path / "a.mp4"
        cause = RuntimeError("no ffprobe")
        with (
            patch(_MEDIA_PROBE, side_effect=cause),
            pytest.raises(ProbeDurationError, match="ffprobe failed") as excinfo,
        ):
            probe_duration(media)
        assert excinfo.value.__cause__ is cause


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
