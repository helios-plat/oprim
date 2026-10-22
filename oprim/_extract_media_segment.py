"""oprim.extract_media_segment — cut a real media file out of a time range.

Single atomic FFmpeg extraction. Produces a standalone playable artifact at the
requested range. Deciding *where* the boundaries belong is not this element's
job — pass the range in (see `segment_media` for boundary detection, and oskill
for semantic segmentation).

Raises:
    ExtractMediaSegmentError: Validation failure or FFmpeg error.
"""

from __future__ import annotations

from pathlib import Path

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run


class ExtractMediaSegmentError(Exception):
    """Segment extraction failed."""


async def extract_media_segment(
    *,
    media_path: Path,
    start: float,
    end: float,
    output_path: Path,
    reencode: bool = False,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    timeout_s: float = 300.0,
) -> Path:
    """Extract the `[start, end)` range of a media file into a new file.

    Args:
        media_path: Source audio or video file.
        start: Range start in seconds (must be >= 0).
        end: Range end in seconds (must be > start).
        output_path: Destination file.
        reencode: False (default) stream-copies for a fast keyframe-aligned cut;
            True re-encodes for frame-accurate boundaries.
        video_codec: Encoder used when `reencode=True`.
        audio_codec: Audio encoder used when `reencode=True`.
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        ExtractMediaSegmentError: Invalid range, missing input, or FFmpeg error.
    """
    if not media_path.exists():
        raise ExtractMediaSegmentError(f"Media file not found: {media_path}")
    if start < 0:
        raise ExtractMediaSegmentError(f"start must be >= 0, got {start}")
    if end <= start:
        raise ExtractMediaSegmentError(f"end must be > start, got start={start} end={end}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExtractMediaSegmentError(
            f"Cannot create parent dir {output_path.parent}: {exc}"
        ) from exc

    args = ["-ss", f"{start:.6f}", "-i", str(media_path), "-t", f"{end - start:.6f}"]
    if reencode:
        args += ["-c:v", video_codec, "-pix_fmt", "yuv420p", "-c:a", audio_codec]
    else:
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    args.append(str(output_path))

    try:
        await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
    except FFmpegError as exc:
        raise ExtractMediaSegmentError(f"FFmpeg segment extraction failed: {exc}") from exc

    return output_path
