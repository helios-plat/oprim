"""oprim.synthesize_speech_chunk — single Kokoro TTS chunk synthesis (atomic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SynthesizeSpeechChunkError(RuntimeError):
    pass


async def synthesize_speech_chunk(
    *,
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
    output_path: str | Path | None = None,
    runtime: Any = None,
) -> Path | bytes:
    """Synthesize a single text chunk to speech via Kokoro.

    Atomic: one model call. Model weights are lazily loaded through
    obase.kokoro_runtime (optional dependency).
    """
    if not text or not text.strip():
        raise SynthesizeSpeechChunkError("text must not be empty")

    if runtime is None:
        try:
            from obase.kokoro_runtime import get_kokoro_runtime
        except Exception as exc:
            raise SynthesizeSpeechChunkError("kokoro runtime unavailable") from exc
        runtime = get_kokoro_runtime()

    try:
        audio = runtime.synthesize(text, voice=voice, speed=speed)
    except Exception as exc:
        raise SynthesizeSpeechChunkError(f"kokoro synthesis failed: {exc}") from exc

    if output_path is not None:
        try:
            from obase.kokoro_runtime import write_audio_file
        except Exception as exc:
            raise SynthesizeSpeechChunkError("kokoro runtime unavailable") from exc
        return write_audio_file(audio, Path(output_path))
    return audio  # type: ignore[return-value]
