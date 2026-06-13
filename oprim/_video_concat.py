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
) -> Path:
    """Concatenate multiple video files into one.

    Args:
        inputs: List of video file paths (≥2 required).
        output_path: Destination file.
        method: 'concat_demuxer' (fast, same codec) or 'concat_filter' (re-encode).
        timeout_s: FFmpeg timeout in seconds.

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

    try:
        if method == "concat_demuxer":
            await _concat_demuxer(inputs, output_path, timeout_s)
        else:
            await _concat_filter(inputs, output_path, timeout_s)
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


async def _concat_filter(inputs: list[Path], output_path: Path, timeout_s: float) -> None:
    n = len(inputs)
    args: list[str] = []
    for p in inputs:
        args.extend(["-i", str(p)])

    filter_parts = "".join(f"[{i}:v][{i}:a]" for i in range(n))
    filter_complex = f"{filter_parts}concat=n={n}:v=1:a=1[outv][outa]"

    args.extend([
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        str(output_path),
    ])
    await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
