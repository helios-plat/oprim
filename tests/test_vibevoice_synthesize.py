"""Tests for oprim.vibevoice_synthesize (M3 — ≥6 tests)."""

from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oprim.vibevoice_synthesize import VibeVoiceError, VibeVoiceSetupError, vibevoice_synthesize


def _make_wav_bytes(n_samples: int = 256, framerate: int = 22050) -> bytes:
    """Create minimal valid WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


@dataclass
class _Line:
    speaker_id: str
    text: str
    voice_ref: Path | None = None


def _make_infer_fn(wav_bytes: bytes | None = None) -> Any:
    """Return a sync callable that produces WAV bytes (used as _inference_fn)."""
    data = wav_bytes or _make_wav_bytes()

    def _infer(text: str, speaker_id: str, voice_ref: Path | None) -> bytes:
        return data

    return _infer


class TestVibevoiceSynthesize:
    async def test_single_speaker_success(self, tmp_path: Path) -> None:
        """Single SpeakerLine → output WAV file produced."""
        out = tmp_path / "single.wav"
        script = [_Line(speaker_id="host", text="Hello world")]
        result = await vibevoice_synthesize(
            script=script,
            output_path=out,
            _inference_fn=_make_infer_fn(),
        )
        assert result == out
        assert out.exists()
        with wave.open(str(out)) as wf:
            assert wf.getnchannels() == 1

    async def test_multi_speaker_four_lines(self, tmp_path: Path) -> None:
        """4 speakers alternating → WAV with all segments concatenated."""
        out = tmp_path / "multi.wav"
        script = [
            _Line(speaker_id=f"s{i}", text=f"line {i}")
            for i in range(4)
        ]
        call_log: list[str] = []

        def _infer(text: str, speaker_id: str, voice_ref: Path | None) -> bytes:
            call_log.append(speaker_id)
            return _make_wav_bytes(128)

        await vibevoice_synthesize(script=script, output_path=out, _inference_fn=_infer)

        assert call_log == ["s0", "s1", "s2", "s3"]
        assert out.exists()

    async def test_voice_ref_passed_to_inference(self, tmp_path: Path) -> None:
        """voice_ref path forwarded to inference function."""
        ref = tmp_path / "ref.wav"
        ref.write_bytes(_make_wav_bytes(512))
        out = tmp_path / "cloned.wav"

        received_ref: list[Path | None] = []

        def _infer(text: str, speaker_id: str, voice_ref: Path | None) -> bytes:
            received_ref.append(voice_ref)
            return _make_wav_bytes()

        script = [_Line(speaker_id="narrator", text="Hello", voice_ref=ref)]
        await vibevoice_synthesize(script=script, output_path=out, _inference_fn=_infer)

        assert received_ref[0] == ref

    async def test_long_script_segmentation_no_oom(self, tmp_path: Path) -> None:
        """Long script (50 lines) completes without error (mock inference)."""
        out = tmp_path / "long.wav"
        script = [_Line(speaker_id="spk", text=f"sentence {i}") for i in range(50)]
        call_count = 0

        def _infer(text: str, speaker_id: str, voice_ref: Path | None) -> bytes:
            nonlocal call_count
            call_count += 1
            return _make_wav_bytes(64)

        await vibevoice_synthesize(script=script, output_path=out, _inference_fn=_infer)

        assert call_count == 50
        assert out.exists()

    async def test_watermark_true_default(self, tmp_path: Path) -> None:
        """watermark=True (default) does not raise and output is produced."""
        out = tmp_path / "wm.wav"
        script = [_Line(speaker_id="s", text="watermarked")]
        await vibevoice_synthesize(
            script=script, output_path=out, watermark=True,
            _inference_fn=_make_infer_fn(),
        )
        assert out.exists()

    async def test_empty_script_raises(self, tmp_path: Path) -> None:
        """Empty script → VibeVoiceError."""
        with pytest.raises(VibeVoiceError, match="empty"):
            await vibevoice_synthesize(
                script=[],
                output_path=tmp_path / "out.wav",
                _inference_fn=_make_infer_fn(),
            )

    async def test_setup_error_on_missing_torch(self, tmp_path: Path) -> None:
        """Missing torch raises VibeVoiceSetupError when no _inference_fn override."""
        import sys
        original = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((VibeVoiceSetupError, ImportError)):
                await vibevoice_synthesize(
                    script=[_Line("s", "hi")],
                    output_path=tmp_path / "out.wav",
                )
        finally:
            if original is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original
