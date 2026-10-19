"""oprim.whisper_transcribe — Whisper transcription (atomic, lazy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WhisperTranscribeError(RuntimeError):
    pass


async def whisper_transcribe(
    *,
    audio_path: str | Path,
    model: str = "base",
    language: str = "auto",
    backend: str = "local",
    model_path: str = "/models/whisper",
) -> Any:
    """Transcribe audio with Whisper (lazy faster-whisper/openai-whisper)."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise WhisperTranscribeError(f"audio file not found: {audio_path}")

    # Reuse existing transcribe_audio when available (single source).
    try:
        from oprim import transcribe_audio  # type: ignore
        return await transcribe_audio(
            audio_path=audio_path,
            backend=backend if backend in ("local", "dashscope") else "local",
            model_size=model,
            language=language if language != "auto" else "zh",
            model_path=model_path,
        )
    except Exception as exc:
        raise WhisperTranscribeError(f"transcription failed: {exc}") from exc


# alias for report naming
whisper_transcribe_audio = whisper_transcribe
