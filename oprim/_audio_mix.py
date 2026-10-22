"""oprim.audio_mix — Multi-track audio mixing via FFmpeg amix filter.

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.audio_mix import audio_mix
    >>> result = asyncio.run(audio_mix(
    ...     inputs=[Path("narration.wav"), Path("bgm.wav")],
    ...     weights=[1.0, 0.3],
    ...     output_path=Path("mixed.wav"),
    ... ))

Raises:
    AudioMixError: Mixing failed (input missing, FFmpeg error, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run


class AudioMixError(Exception):
    """Audio mixing failed."""


async def audio_mix(
    *,
    inputs: list[Path],
    weights: list[float] | None = None,
    output_path: Path,
    sample_rate: int = 44100,
    offsets: list[float] | None = None,
    fade_in_s: float | list[float] | None = None,
    fade_out_s: float | list[float] | None = None,
    duration: Literal["longest", "shortest", "first"] = "longest",
    timeout_s: float = 120.0,
) -> Path:
    """Mix multiple audio tracks into one output file.

    Args:
        inputs: List of audio file paths to mix.
        weights: Volume weight per track (0.0–1.0). Defaults to 1.0 each.
        output_path: Destination file path.
        sample_rate: Output sample rate in Hz.
        offsets: Per-track start delay in seconds (negative rejected). Defaults
            to no delay for every track.
        fade_in_s: Per-track fade-in duration in seconds, or one value applied
            to all tracks. ``None`` disables fade-in.
        fade_out_s: Per-track fade-out duration in seconds, or one value applied
            to all tracks. ``None`` disables fade-out.
        duration: Duration policy for `amix` — "longest" (default), "shortest",
            or "first" (length of the first input).
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        AudioMixError: On validation failure or FFmpeg error.

    Example:
        >>> await audio_mix(inputs=[Path("a.wav"), Path("b.wav")], output_path=Path("out.wav"))
    """
    if not inputs:
        raise AudioMixError("inputs must not be empty")

    for p in inputs:
        if not p.exists():
            raise AudioMixError(f"Input file not found: {p}")

    n = len(inputs)

    if weights is None:
        weights = [1.0] * n

    if len(weights) != n:
        raise AudioMixError("weights length must match inputs length")

    if offsets is None:
        offsets = [0.0] * n
    elif len(offsets) != n:
        raise AudioMixError("offsets length must match inputs length")
    elif any(o < 0 for o in offsets):
        raise AudioMixError("offsets must be >= 0 (use atrim for negative offsets)")

    fade_in = _per_track(fade_in_s, n, "fade_in_s")
    fade_out = _per_track(fade_out_s, n, "fade_out_s")

    if duration not in ("longest", "shortest", "first"):
        raise AudioMixError(f"Unknown duration policy: {duration!r}")

    args: list[str] = []
    for p in inputs:
        args.extend(["-i", str(p)])

    filter_complex = ";".join(
        _track_chain(
            i, weight=weights[i], offset=offsets[i], fade_in=fade_in[i], fade_out=fade_out[i]
        )
        for i in range(n)
    )
    mix_inputs = "".join(f"[a{i}]" for i in range(n))
    filter_complex += f";{mix_inputs}amix=inputs={n}:duration={duration}"

    args.extend(
        [
            "-filter_complex",
            filter_complex,
            "-ar",
            str(sample_rate),
            str(output_path),
        ]
    )

    try:
        await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
    except FFmpegError as exc:
        raise AudioMixError(f"FFmpeg mixing failed: {exc}") from exc

    return output_path


def _per_track(value: float | list[float] | None, n: int, name: str) -> list[float]:
    """Normalize a scalar-or-per-track option into an n-length list."""
    if value is None:
        return [0.0] * n
    values = [float(value)] * n if isinstance(value, (int, float)) else [float(v) for v in value]
    if len(values) != n:
        raise AudioMixError(f"{name} length must match inputs length")
    if any(v < 0 for v in values):
        raise AudioMixError(f"{name} must be >= 0")
    return values


def _track_chain(
    index: int, *, weight: float, offset: float, fade_in: float, fade_out: float
) -> str:
    """Build one `[i]<filters>[a{i}]` chain: volume → adelay → fades."""
    filters = [f"volume={weight}"]
    if offset > 0:
        filters.append(f"adelay=delays={int(round(offset * 1000))}:all=1")
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        filters.append(f"afade=t=out:st=-{fade_out}:d={fade_out}")
    return f"[{index}]{','.join(filters)}[a{index}]"


async def mix_audio_tracks(
    *,
    inputs: list[Path],
    weights: list[float] | None = None,
    output_path: Path,
    sample_rate: int = 44100,
    offsets: list[float] | None = None,
    fade_in_s: float | list[float] | None = None,
    fade_out_s: float | list[float] | None = None,
    duration: Literal["longest", "shortest", "first"] = "longest",
    timeout_s: float = 120.0,
) -> Path:
    """Capability-oriented alias for `audio_mix` (SPEC §4.8 canonical name).

    Same-module alias: both names resolve to one implementation, so mixing has
    exactly one canonical element.
    """
    return await audio_mix(
        inputs=inputs,
        weights=weights,
        output_path=output_path,
        sample_rate=sample_rate,
        offsets=offsets,
        fade_in_s=fade_in_s,
        fade_out_s=fade_out_s,
        duration=duration,
        timeout_s=timeout_s,
    )
