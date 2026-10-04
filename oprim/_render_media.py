"""oprim.render_media — general frame-sequence → MP4 render atom.

The renderer-agnostic render path (SPEC §4.7). `render_html_to_mp4` remains the
HTML/CSS specialization; both encode through the single `_encode_frames`
implementation, so there is one canonical encode step and no element calls
another element.

Raises:
    RenderMediaError: Validation failure or encode failure.
"""

from __future__ import annotations

from pathlib import Path

from oprim._encode_frames import EncodeFramesError, encode_frames_to_mp4


class RenderMediaError(Exception):
    """Media render failed."""


async def render_media(
    *,
    frames_pattern: str,
    output_path: Path,
    fps: int = 30,
    frame_count: int | None = None,
    width: int | None = None,
    height: int | None = None,
    timeout_s: float = 300.0,
) -> Path:
    """Render a numbered image sequence into an H.264 MP4.

    Args:
        frames_pattern: ffmpeg input pattern, e.g. `/tmp/frames/frame_%06d.png`.
        output_path: Destination MP4 file.
        fps: Frames per second of the source sequence.
        frame_count: Optional expected frame count; validated against the
            pattern's directory listing to fail fast on an incomplete render.
        width, height: Optional exact output scaling.
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        The output_path on success.

    Raises:
        RenderMediaError: Bad arguments, frame count mismatch, or encode failure.
    """
    if not frames_pattern:
        raise RenderMediaError("frames_pattern is required")
    if fps <= 0:
        raise RenderMediaError("fps must be > 0")
    if width is not None and width <= 0:
        raise RenderMediaError("width must be > 0 when provided")
    if height is not None and height <= 0:
        raise RenderMediaError("height must be > 0 when provided")

    if frame_count is not None:
        if frame_count <= 0:
            raise RenderMediaError("frame_count must be > 0 when provided")
        available = _count_frames(frames_pattern)
        if available == 0:
            raise RenderMediaError(f"No frames match pattern: {frames_pattern}")
        if available < frame_count:
            raise RenderMediaError(
                f"Incomplete frame sequence: pattern {frames_pattern} yielded "
                f"{available} frames, expected {frame_count}"
            )

    try:
        return await encode_frames_to_mp4(
            input_pattern=frames_pattern,
            output_path=output_path,
            fps=fps,
            width=width,
            height=height,
            timeout_s=timeout_s,
        )
    except EncodeFramesError as exc:
        raise RenderMediaError(str(exc)) from exc


def _count_frames(frames_pattern: str) -> int:
    """Glob the directory behind an ffmpeg `%0Nd` pattern and count matches."""
    directory, _, name = frames_pattern.rpartition("/")
    if not name:
        return 0
    stem = name.split("%0")[0]
    suffix = name.rsplit(".", 1)[-1] if "." in name else ""
    pattern = f"{stem}*{suffix}" if suffix else f"{stem}*"
    base = Path(directory) if directory else Path(".")
    try:
        return sum(1 for entry in base.glob(pattern) if entry.is_file())
    except OSError:
        return 0
