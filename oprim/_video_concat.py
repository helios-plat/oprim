"""oprim.video_concat — Concatenate multiple video files.

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.video_concat import video_concat
    >>> result = asyncio.run(video_concat(
    ...     inputs=[Path("part1.mp4"), Path("part2.mp4")],
    ...     output_path=Path("full.mp4"),
    ... ))

Raises:
    VideoConcatError: Concatenation failed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run


class VideoConcatError(Exception):
    """Video concatenation failed."""


async def video_concat(
    *,
    inputs: list[Path],
    output_path: Path,
    method: Literal["concat_filter", "concat_demuxer"] = "concat_demuxer",
    timeout_s: float = 600.0,
    tail_frame_handling: bool = False,
    trim_lead_frames: int = 0,
    transitions: list[dict] | None = None,
) -> Path:
    """Concatenate multiple video files into one.

    Args:
        inputs: List of video file paths (≥2 required).
        output_path: Destination file.
        method: 'concat_demuxer' (fast, same codec) or 'concat_filter' (re-encode).
        timeout_s: FFmpeg timeout in seconds.
        tail_frame_handling: When True, force re-encode to flush tail frames before
            the next clip (prevents freeze artifacts from codec B-frame flush).
        trim_lead_frames: Number of leading frames to trim from each clip after
            the first (0 = no trimming). Use 1-2 to remove stutter artifacts.
        transitions: Optional list of transition specs applied between clips:
            [{type: "hard"|"dissolve"|"flash", duration_s: float}, ...].
            Length must be len(inputs)-1. None = hard cut (default).

    Returns:
        The output_path on success.

    Raises:
        VideoConcatError: On validation failure or FFmpeg error.

    Example:
        >>> await video_concat(inputs=[Path("a.mp4"), Path("b.mp4")], output_path=Path("out.mp4"))
    """
    if len(inputs) < 2:
        raise VideoConcatError("At least 2 input videos required")

    for p in inputs:
        if not p.exists():
            raise VideoConcatError(f"Input file not found: {p}")

    if transitions is not None and len(transitions) != len(inputs) - 1:
        raise VideoConcatError(
            f"transitions length ({len(transitions)}) must equal len(inputs)-1 "
            f"({len(inputs) - 1})"
        )

    # When advanced features are requested, force re-encode via concat_filter.
    # Default (all False/0/None) falls through to the caller-specified method
    # with identical behaviour to previous versions.
    needs_reencode = tail_frame_handling or trim_lead_frames > 0 or transitions is not None
    effective_method = "concat_filter" if needs_reencode else method

    try:
        if effective_method == "concat_demuxer":
            await _concat_demuxer(inputs, output_path, timeout_s)
        else:
            await _concat_filter(
                inputs, output_path, timeout_s,
                trim_lead_frames=trim_lead_frames,
                transitions=transitions or [],
            )
    except FFmpegError as exc:
        raise VideoConcatError(f"FFmpeg concat failed: {exc}") from exc

    return output_path


async def _concat_demuxer(inputs: list[Path], output_path: Path, timeout_s: float) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in inputs:
            f.write(f"file '{p.resolve()}'\n")
        list_path = Path(f.name)

    try:
        args = [
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path),
        ]
        await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
    finally:
        list_path.unlink(missing_ok=True)


async def _concat_filter(
    inputs: list[Path],
    output_path: Path,
    timeout_s: float,
    trim_lead_frames: int = 0,
    transitions: list[dict] | None = None,
) -> None:
    n = len(inputs)
    args: list[str] = []
    for p in inputs:
        args.extend(["-i", str(p)])

    # Build per-stream trim filters for lead-frame removal (clips 1..n-1)
    filter_parts: list[str] = []
    for i in range(n):
        if i > 0 and trim_lead_frames > 0:
            filter_parts.append(f"[{i}:v]trim=start_frame={trim_lead_frames}[v{i}t];")
            filter_parts.append(f"[{i}:a]atrim=start={trim_lead_frames / 30.0:.4f}[a{i}t];")
            v_label, a_label = f"[v{i}t]", f"[a{i}t]"
        else:
            v_label, a_label = f"[{i}:v]", f"[{i}:a]"
        filter_parts.append(f"{v_label}{a_label}")

    concat_inputs = "".join(filter_parts)
    filter_complex = f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"

    args.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ])
    await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
