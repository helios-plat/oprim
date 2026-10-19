"""oprim.extract_audio_waveform — decode an audio file to a normalized waveform.

Single atomic decode pass. Produces the peak/RMS envelope plus downsampled
samples that timeline editors, QC checks, and level meters consume. Emits raw
PCM to a null sink and reads the levels ffmpeg reports on stderr — no numpy
required, so this stays a thin atomic element.

Raises:
    ExtractAudioWaveformError: Validation failure or FFmpeg error.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

from oprim._exceptions import OprimError

_VOLUME_RE = re.compile(
    r"mean_volume:\s*(?P<mean>-?[0-9.]+)\s*dB.*?max_volume:\s*(?P<peak>-?[0-9.]+)\s*dB", re.DOTALL
)
_DEFAULT_TIMEOUT_S = 120.0


class ExtractAudioWaveformError(OprimError):
    """Waveform extraction failed."""


class AudioWaveform(BaseModel):
    source: Path
    sample_rate: int
    duration: float
    peak_db: float | None = None
    rms_db: float | None = None
    samples: list[float] = []


def _parse_volume(stderr: str) -> tuple[float | None, float | None]:
    match = None
    for candidate in _VOLUME_RE.finditer(stderr):
        match = candidate  # noqa: B007 — keep the final reading ffmpeg emits
    if match is None:
        return None, None
    return float(match.group("mean")), float(match.group("peak"))


def _decode_pcm(path: Path, *, sample_rate: int, timeout_s: float) -> bytes:
    """Decode to mono s16le at `sample_rate` and return the raw PCM bytes."""
    if shutil.which("ffmpeg") is None:
        raise ExtractAudioWaveformError("ffmpeg not found on PATH")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s, check=True  # noqa: S603
        )
    except subprocess.TimeoutExpired as exc:
        raise ExtractAudioWaveformError(f"ffmpeg timed out on {path}", cause=exc) from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ExtractAudioWaveformError(
            f"ffmpeg decode failed for {path}: {exc}", cause=exc
        ) from exc
    return proc.stdout


def _downsample(pcm: bytes, *, target_samples: int) -> list[float]:
    """Peak-normalize s16le PCM and reduce to at most `target_samples` peaks."""
    count = len(pcm) // 2
    if count == 0:
        return []
    values = [int.from_bytes(pcm[i * 2 : i * 2 + 2], "little", signed=True) for i in range(count)]
    absolute_max = max(abs(v) for v in values)
    if absolute_max == 0:
        return [0.0] * min(target_samples, count)
    if count <= target_samples:
        return [round(v / absolute_max, 6) for v in values]
    bucket = count / target_samples
    peaks: list[float] = []
    for i in range(target_samples):
        lo = int(i * bucket)
        hi = max(lo + 1, int((i + 1) * bucket))
        peaks.append(round(max(abs(v) for v in values[lo:hi]) / absolute_max, 6))
    return peaks


def extract_audio_waveform(
    *,
    audio_path: Path,
    sample_rate: int = 8000,
    target_samples: int = 1000,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> AudioWaveform:
    """Decode an audio file into a normalized waveform envelope.

    Args:
        audio_path: Source audio file (any container ffmpeg can decode).
        sample_rate: Analysis sample rate in Hz. 8000 is ample for envelopes.
        target_samples: Maximum number of samples in the returned envelope.
        timeout_s: Per-pass ffmpeg timeout in seconds.

    Returns:
        AudioWaveform with sample_rate, duration, peak/rms dB, and normalized
        samples in [-1.0, 1.0].

    Raises:
        ExtractAudioWaveformError: Missing input, invalid arguments, or
            ffmpeg failure.
    """
    if not audio_path.exists():
        raise ExtractAudioWaveformError(f"Audio file not found: {audio_path}")
    if sample_rate <= 0:
        raise ExtractAudioWaveformError("sample_rate must be > 0")
    if target_samples <= 0:
        raise ExtractAudioWaveformError("target_samples must be > 0")

    pcm = _decode_pcm(audio_path, sample_rate=sample_rate, timeout_s=timeout_s)
    frame_count = len(pcm) // 2
    duration = frame_count / sample_rate if sample_rate else 0.0

    stderr = ""
    if shutil.which("ffmpeg") is not None:
        try:
            proc = subprocess.run(  # noqa: S603
                [
                    "ffmpeg",
                    "-v",
                    "info",
                    "-i",
                    str(audio_path),
                    "-af",
                    "volumedetect",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            stderr = proc.stderr
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ExtractAudioWaveformError(
                f"ffmpeg volumedetect failed: {exc}", cause=exc
            ) from exc

    rms_db, peak_db = _parse_volume(stderr)
    return AudioWaveform(
        source=audio_path,
        sample_rate=sample_rate,
        duration=round(duration, 6),
        peak_db=peak_db,
        rms_db=rms_db,
        samples=_downsample(pcm, target_samples=target_samples),
    )
