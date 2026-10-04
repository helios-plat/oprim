"""oprim.extract_video_frames — extract still frames from a video at chosen positions.

One atomic extraction. No analysis: the caller supplies which frames it wants
(`fps`, `timestamps`, `count`, or `scene_boundaries`) or this element reports
what it wrote. Deciding *why* a frame matters belongs to oskill.

Selection modes are mutually exclusive:

* ``fps``            — uniform sampling at N frames per second (single pass).
* ``timestamps``     — explicit ``(start, end)`` second positions.
* ``count``          — N frames spread across ``duration_s`` (requires it).
* ``scene_boundaries`` — caller-computed boundary hints, same shape as timestamps.

Raises:
    ExtractVideoFramesError: Validation failure or FFmpeg error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run
from pydantic import BaseModel

_SAFE_INDEX = re.compile(r"[^0-9a-zA-Z_.-]")


class ExtractVideoFramesError(Exception):
    """Frame extraction failed."""


class ExtractedFrame(BaseModel):
    index: int
    timestamp: float
    path: Path


class ExtractedFrames(BaseModel):
    source: Path
    width: int | None = None
    height: int | None = None
    frames: list[ExtractedFrame] = []


def _sanitize(value: str) -> str:
    return _SAFE_INDEX.sub("_", value)


def _single_frame_args(
    *,
    video: Path,
    timestamp: float,
    output_path: Path,
    width: int | None,
    height: int | None,
) -> list[str]:
    vf = ["-frames:v", "1"]
    if width and height:
        vf = ["-vf", f"scale={width}:{height}", *vf]
    return [
        "-ss",
        f"{max(0.0, timestamp):.6f}",
        "-i",
        str(video),
        *vf,
        "-q:v",
        "2",
        str(output_path),
    ]


async def extract_video_frames(
    *,
    video_path: Path,
    output_dir: Path,
    mode: Literal["fps", "timestamps", "count", "scene_boundaries"] = "fps",
    fps: float | None = None,
    timestamps: list[float] | None = None,
    scene_boundaries: list[float] | None = None,
    count: int | None = None,
    duration_s: float | None = None,
    width: int | None = None,
    height: int | None = None,
    timeout_s: float = 120.0,
) -> ExtractedFrames:
    """Extract still frames from a video.

    Args:
        video_path: Source video file.
        output_dir: Directory to write frames into (created if absent).
        mode: Which selection strategy to use. Exactly one of the selector
            arguments matching `mode` must be supplied.
        fps: Frames per second for `mode="fps"`.
        timestamps: Explicit second positions for `mode="timestamps"`.
        scene_boundaries: Boundary hints for `mode="scene_boundaries"`; treated
            as explicit positions (this element does not detect boundaries).
        count: Number of frames for `mode="count"`, spread over `duration_s`.
        duration_s: Required with `count`; total source duration in seconds.
        width, height: Optional output scaling.
        timeout_s: Per-invocation FFmpeg timeout.

    Returns:
        ExtractedFrames with the written frame paths and their timestamps.

    Raises:
        ExtractVideoFramesError: Invalid selector combination, missing input,
            or FFmpeg failure.
    """
    if not video_path.exists():
        raise ExtractVideoFramesError(f"Video file not found: {video_path}")

    supplied = {
        "fps": fps is not None,
        "timestamps": timestamps is not None,
        "scene_boundaries": scene_boundaries is not None,
        "count": count is not None,
    }
    active = [name for name, given in supplied.items() if given]
    if mode not in supplied:
        raise ExtractVideoFramesError(f"Unknown mode: {mode!r}")
    if not supplied[mode]:
        raise ExtractVideoFramesError(f"mode={mode!r} requires its selector argument")
    if len(active) > 1:
        raise ExtractVideoFramesError(
            f"Exactly one selector allowed, got {active}; drop mode= and keep one"
        )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExtractVideoFramesError(f"Cannot create output_dir {output_dir}: {exc}") from exc

    stem = _sanitize(video_path.stem)

    try:
        if mode == "fps":
            return await _extract_uniform(
                video_path=video_path,
                output_dir=output_dir,
                stem=stem,
                fps=float(fps or 0.0),
                width=width,
                height=height,
                timeout_s=timeout_s,
            )
        if mode == "count":
            return await _extract_spread(
                video_path=video_path,
                output_dir=output_dir,
                stem=stem,
                count=int(count or 0),
                duration_s=duration_s,
                width=width,
                height=height,
                timeout_s=timeout_s,
            )
        positions = timestamps if mode == "timestamps" else scene_boundaries
        return await _extract_at(
            video_path=video_path,
            output_dir=output_dir,
            stem=stem,
            positions=list(positions or []),
            width=width,
            height=height,
            timeout_s=timeout_s,
        )
    except FFmpegError as exc:
        raise ExtractVideoFramesError(f"FFmpeg frame extraction failed: {exc}") from exc


async def _extract_uniform(
    *,
    video_path: Path,
    output_dir: Path,
    stem: str,
    fps: float,
    width: int | None,
    height: int | None,
    timeout_s: float,
) -> ExtractedFrames:
    if fps <= 0:
        raise ExtractVideoFramesError("fps must be > 0")
    pattern = output_dir / f"{stem}_f%06d.jpg"
    vf = f"fps={fps}"
    if width and height:
        vf = f"{vf},scale={width}:{height}"
    args = ["-i", str(video_path), "-vf", vf, "-q:v", "2", str(pattern)]
    await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_dir)

    written = sorted(output_dir.glob(f"{stem}_f*.jpg"))
    if not written:
        raise ExtractVideoFramesError(f"No frames written for {video_path}")
    frames = [
        ExtractedFrame(index=i, timestamp=round(i / fps, 6), path=p) for i, p in enumerate(written)
    ]
    return ExtractedFrames(
        source=video_path, width=width, height=height, frames=frames
    )


async def _extract_spread(
    *,
    video_path: Path,
    output_dir: Path,
    stem: str,
    count: int,
    duration_s: float | None,
    width: int | None,
    height: int | None,
    timeout_s: float,
) -> ExtractedFrames:
    if count <= 0:
        raise ExtractVideoFramesError("count must be > 0")
    if duration_s is None or duration_s <= 0:
        raise ExtractVideoFramesError("mode='count' requires a positive duration_s")
    # Sample at interval midpoints so no frame lands on the exclusive tail.
    step = duration_s / count
    positions = [round((i + 0.5) * step, 6) for i in range(count)]
    return await _extract_at(
        video_path=video_path,
        output_dir=output_dir,
        stem=stem,
        positions=positions,
        width=width,
        height=height,
        timeout_s=timeout_s,
    )


async def _extract_at(
    *,
    video_path: Path,
    output_dir: Path,
    stem: str,
    positions: list[float],
    width: int | None,
    height: int | None,
    timeout_s: float,
) -> ExtractedFrames:
    if not positions:
        raise ExtractVideoFramesError("positions must not be empty")
    frames: list[ExtractedFrame] = []
    for i, ts in enumerate(positions):
        if ts < 0:
            raise ExtractVideoFramesError(f"timestamp must be >= 0, got {ts}")
        target = output_dir / f"{stem}_t{i:06d}.jpg"
        await ffmpeg_run(
            args=_single_frame_args(
                video=video_path,
                timestamp=ts,
                output_path=target,
                width=width,
                height=height,
            ),
            timeout_s=timeout_s,
            expected_output=target,
        )
        frames.append(ExtractedFrame(index=i, timestamp=round(float(ts), 6), path=target))
    return ExtractedFrames(source=video_path, width=width, height=height, frames=frames)
