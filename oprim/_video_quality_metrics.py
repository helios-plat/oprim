"""oprim.video_quality_metrics — Extract technical video quality metrics via ffprobe.

Example:
    >>> from oprim.video_quality_metrics import video_quality_metrics
    >>> m = await video_quality_metrics(video_path=Path("video.mp4"))

Raises:
    VideoQualityError: Extraction failed.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run
from pydantic import BaseModel

#: ffmpeg volumedetect reports e.g. "mean_volume: -21.3 dB"
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[0-9.]+)\s*dB")


class VideoQualityError(Exception):
    """Video quality metrics extraction failed."""


class VideoQualityMetrics(BaseModel):
    """Technical video quality metrics."""

    width: int
    height: int
    duration_s: float
    fps: float
    bitrate_kbps: int
    audio_lufs: float | None = None
    codec_video: str
    codec_audio: str | None = None


async def video_quality_metrics(
    *, video_path: Path, measure_loudness: bool = False
) -> VideoQualityMetrics:
    """Extract technical quality metrics from a video file.

    Args:
        video_path: Path to the video file.
        measure_loudness: When True and the file has an audio stream, run an
            ffmpeg `volumedetect` pass and report integrated loudness in
            `audio_lufs`. Off by default so the common call stays a single
            ffprobe pass; that field used to be declared but never written,
            which made it a silent lie.
        video_path: Path to video file.

    Returns:
        VideoQualityMetrics model.

    Raises:
        VideoQualityError: File not found, ffprobe missing, or parse failure.

    Example:
        >>> m = await video_quality_metrics(video_path=Path("v.mp4"))
    """
    if not video_path.exists():
        raise VideoQualityError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except FileNotFoundError:
        raise VideoQualityError("ffprobe not found on PATH") from None

    if proc.returncode != 0:
        raise VideoQualityError("ffprobe failed")

    data = json.loads(stdout.decode())
    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if not video_stream:
        raise VideoQualityError("No video stream found")

    fps_str = video_stream.get("r_frame_rate", "25/1")
    num, den = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
    fps = float(num) / float(den) if float(den) != 0 else 25.0

    lufs: float | None = None
    if measure_loudness and audio_stream is not None:
        lufs = await _measure_lufs(video_path)

    return VideoQualityMetrics(
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        duration_s=float(fmt.get("duration", 0)),
        fps=fps,
        bitrate_kbps=int(float(fmt.get("bit_rate", 0)) / 1000),
        audio_lufs=lufs,
        codec_video=video_stream.get("codec_name", "unknown"),
        codec_audio=audio_stream.get("codec_name") if audio_stream else None,
    )


async def _measure_lufs(video_path: Path) -> float | None:
    """Integrated loudness via ffmpeg volumedetect; None when unavailable.

    Loudness measurement is optional QC, never a reason to fail the metrics call,
    so FFmpeg errors degrade to None instead of propagating.
    """
    try:
        stderr = await ffmpeg_run(
            args=["-i", str(video_path), "-af", "volumedetect", "-f", "null", "-"],
            timeout_s=120.0,
        )
    except FFmpegError:
        return None
    match = _MEAN_VOLUME_RE.search(stderr)
    return float(match.group(1)) if match else None
