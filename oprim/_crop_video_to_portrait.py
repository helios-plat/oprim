"""oprim.crop_video_to_portrait — crop video to 9:16 portrait (atomic, ffmpeg)."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


class CropVideoToPortraitError(RuntimeError):
    pass


async def crop_video_to_portrait(
    *,
    input_path: str | Path,
    output_path: str | Path | None = None,
    width: int = 1080,
    height: int = 1920,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Crop a video to portrait 9:16 using ffmpeg (center crop + scale)."""
    inp = Path(input_path)
    if not inp.exists():
        raise CropVideoToPortraitError(f"input not found: {inp}")
    if shutil.which(ffmpeg) is None:
        raise CropVideoToPortraitError("ffmpeg not found on PATH")

    if output_path is None:
        output_path = inp.with_name(inp.stem + "_portrait.mp4")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # center crop to 9:16 then scale
    vf = (
        f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={width}:{height}"
    )
    cmd = [
        ffmpeg, "-y", "-i", str(inp),
        "-vf", vf,
        "-c:a", "copy",
        str(out),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise CropVideoToPortraitError(f"ffmpeg failed: {err.decode(errors='replace')[:500]}")
    return out
