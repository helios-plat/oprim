"""Tests for oprim.edge_tts_synthesize (多语言云 TTS — ≥6 tests).

用注入的 _synth_fn / _ffmpeg_fn 避开 edge_tts / obase.ffmpeg 依赖(不触网、不装可选依赖)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from oprim import EdgeTtsError, edge_tts_synthesize
from oprim._edge_tts_synthesize import _voice_for


@dataclass
class _Line:
    text: str


def _writer_synth(payload: bytes = b"\xff\xf3ID3fakeaudio"):
    async def _synth(text: str, voice: str, seg: Path) -> None:
        seg.write_bytes(payload)

    return _synth


async def _failing_synth(text: str, voice: str, seg: Path) -> None:
    raise RuntimeError("edge-tts unavailable")


def _capturing_ffmpeg(sink: dict):
    async def _ff(*, args: list[str], expected_output: Path) -> None:
        sink["args"] = args
        expected_output.write_bytes(b"RIFFwavdata")

    return _ff


class TestVoiceSelection:
    def test_cjk_autoselects_zh(self) -> None:
        assert _voice_for("你好世界", None).startswith("zh-")

    def test_ascii_autoselects_en(self) -> None:
        assert _voice_for("hello there", None).startswith("en-")

    def test_explicit_language_overrides(self) -> None:
        assert _voice_for("hello", "zh").startswith("zh-")  # 显式中文即便文本是英文
        assert _voice_for("你好", "en").startswith("en-")
        assert _voice_for("x", "ja") == "ja-JP-NanamiNeural"


class TestEdgeTtsSynthesize:
    async def test_empty_script_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            await edge_tts_synthesize(
                script=[], output_path=tmp_path / "o.wav", _synth_fn=_writer_synth()
            )

    async def test_all_segments_fail_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EdgeTtsError, match="all segments failed"):
            await edge_tts_synthesize(
                script=[_Line("a"), _Line("b")],
                output_path=tmp_path / "o.wav",
                _synth_fn=_failing_synth,
                _ffmpeg_fn=_capturing_ffmpeg({}),
            )

    async def test_single_line_transcodes(self, tmp_path: Path) -> None:
        sink: dict = {}
        out = tmp_path / "o.wav"
        res = await edge_tts_synthesize(
            script=[_Line("你好")],
            output_path=out,
            _synth_fn=_writer_synth(),
            _ffmpeg_fn=_capturing_ffmpeg(sink),
        )
        assert res == out
        assert out.exists()
        assert "-filter_complex" not in sink["args"]  # 单段:直接转码,不 concat
        assert sink["args"][-3:] == ["-ac", "1", str(out)]

    async def test_multi_line_uses_concat(self, tmp_path: Path) -> None:
        sink: dict = {}
        await edge_tts_synthesize(
            script=[_Line("一"), _Line("二"), _Line("三")],
            output_path=tmp_path / "o.wav",
            _synth_fn=_writer_synth(),
            _ffmpeg_fn=_capturing_ffmpeg(sink),
        )
        assert "-filter_complex" in sink["args"]
        assert any("concat=n=3:v=0:a=1[a]" in a for a in sink["args"])

    async def test_partial_failure_still_succeeds(self, tmp_path: Path) -> None:
        calls = {"n": 0}

        async def _flaky(text: str, voice: str, seg: Path) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first line dropped")
            seg.write_bytes(b"\xff\xf3audio")

        out = tmp_path / "o.wav"
        res = await edge_tts_synthesize(
            script=[_Line("drop me"), _Line("keep me")],
            output_path=out,
            _synth_fn=_flaky,
            _ffmpeg_fn=_capturing_ffmpeg({}),
        )
        assert res == out and out.exists()

    async def test_language_from_config(self, tmp_path: Path) -> None:
        """language 缺省时取自 config['language'];此处仅验证不报错并出片。"""
        out = tmp_path / "o.wav"
        res = await edge_tts_synthesize(
            config={"language": "en"},
            script=[_Line("hello")],
            output_path=out,
            _synth_fn=_writer_synth(),
            _ffmpeg_fn=_capturing_ffmpeg({}),
        )
        assert res == out
