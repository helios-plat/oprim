"""oprim.probe_duration — ffprobe 实测媒体文件时长(秒)。"""

from __future__ import annotations

import subprocess
from pathlib import Path


class ProbeDurationError(Exception):
    """ffprobe duration probing failed."""


def probe_duration(path: Path | str) -> float:
    """返回媒体文件真实时长(秒);探测失败抛 ProbeDurationError。"""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise ProbeDurationError(f"ffprobe failed for {path}: {exc}") from exc
    try:
        return float(out.stdout.strip())
    except ValueError as exc:
        raise ProbeDurationError(f"unparsable ffprobe duration for {path}: {out.stdout!r}") from exc
