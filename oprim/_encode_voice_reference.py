"""oprim.encode_voice_reference — canonicalize a reference clip into a reusable voice asset.

Single atomic transform: reference audio in, normalized voice-reference artifact
out. Produces the provider-agnostic form every cloning TTS expects (mono, fixed
sample rate, loudness-normalized, trimmed) plus a content-addressed id so the
same reference can be cached and reused across calls.

This element does *not* synthesize speech and does not embed a voice model —
embedding extraction belongs to a provider. It only guarantees the reference is
in a shape a provider can accept.

Raises:
    EncodeVoiceReferenceError: Validation failure or FFmpeg error.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run
from pydantic import BaseModel


class EncodeVoiceReferenceError(Exception):
    """Voice reference encoding failed."""


class VoiceReference(BaseModel):
    """A reusable, provider-agnostic voice reference artifact."""

    reference_id: str
    source: Path
    path: Path
    sample_rate: int
    channels: int
    max_duration_s: float
    sha256: str


async def encode_voice_reference(
    *,
    audio_path: Path,
    output_path: Path,
    sample_rate: int = 24000,
    max_duration_s: float = 15.0,
    target_lufs: float = -20.0,
    timeout_s: float = 120.0,
) -> VoiceReference:
    """Normalize a reference clip into a reusable voice-reference artifact.

    Args:
        audio_path: Source reference recording (speech sample).
        output_path: Destination WAV path.
        sample_rate: Output sample rate in Hz. 24000 suits most cloning models.
        max_duration_s: Trim the reference to at most this many seconds.
        target_lufs: Integrated loudness target for EBU R128 normalization.
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        VoiceReference describing the written artifact, including a
        content-addressed `reference_id` derived from the source bytes.

    Raises:
        EncodeVoiceReferenceError: Missing input, invalid arguments, or
            FFmpeg failure.
    """
    if not audio_path.exists():
        raise EncodeVoiceReferenceError(f"Reference audio not found: {audio_path}")
    if sample_rate <= 0:
        raise EncodeVoiceReferenceError("sample_rate must be > 0")
    if max_duration_s <= 0:
        raise EncodeVoiceReferenceError("max_duration_s must be > 0")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EncodeVoiceReferenceError(
            f"Cannot create parent dir {output_path.parent}: {exc}"
        ) from exc

    digest = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    reference_id = f"vref_{digest[:24]}"

    args = [
        "-i",
        str(audio_path),
        "-t",
        f"{max_duration_s:.6f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-2.0:LRA=11",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    try:
        await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
    except FFmpegError as exc:
        raise EncodeVoiceReferenceError(f"FFmpeg voice reference encode failed: {exc}") from exc

    return VoiceReference(
        reference_id=reference_id,
        source=audio_path,
        path=output_path,
        sample_rate=sample_rate,
        channels=1,
        max_duration_s=max_duration_s,
        sha256=digest,
    )
