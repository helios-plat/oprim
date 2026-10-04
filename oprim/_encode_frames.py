"""oprim._encode_frames — canonical image-sequence → H.264/MP4 encoder.

Private infra module. One implementation of the encode step, shared by the
`render_media` and `render_html_to_mp4` atoms so the two public elements cannot
drift apart (SPEC §4.7, §8.4). Not an element: it encodes nothing on its own
without a caller-owned frame source.

Output codec is pinned to match `video_generate` so results stay directly
assemblable by `video_concat`.
"""

from __future__ import annotations

from pathlib import Path

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run


class EncodeFramesError(Exception):
    """Frame encoding failed."""


async def encode_frames_to_mp4(
    *,
    input_pattern: str,
    output_path: Path,
    fps: int,
    width: int | None = None,
    height: int | None = None,
    timeout_s: float = 300.0,
) -> Path:
    """Encode a numbered image sequence to H.264/AAC MP4.

    Args:
        input_pattern: ffmpeg input pattern, e.g. `/tmp/frames/frame_%06d.png`.
        output_path: Destination MP4 file.
        fps: Frames per second of the source sequence.
        width, height: Optional exact output scaling.
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        EncodeFramesError: FFmpeg failure or missing output.
    """
    if fps <= 0:
        raise EncodeFramesError("fps must be > 0")

    args = ["-framerate", str(fps), "-i", input_pattern, "-c:v", "libx264"]
    if width and height:
        args += ["-vf", f"scale={width}:{height}"]
    args += [
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        await ffmpeg_run(args=args, timeout_s=timeout_s, expected_output=output_path)
    except FFmpegError as exc:
        raise EncodeFramesError(f"FFmpeg frame encode failed: {exc}") from exc
    return output_path
