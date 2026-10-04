"""oprim._ffprobe — canonical ffprobe executor shared by every media inspection atom.

Single source of truth for ffprobe binary resolution, timeout policy, output
decoding, and the ``FFprobeError`` taxonomy. Public atoms (``media_probe``,
``probe_duration``, ``segment_media``, ``extract_video_frames``,
``extract_media_segment``) never spawn ffprobe themselves.

Private infra module — same tier as ``_exceptions`` / ``_config``. Not an
oprim element: it performs no media operation on its own.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from oprim._exceptions import OprimError

DEFAULT_TIMEOUT_S = 30.0


class FFprobeError(OprimError):
    """ffprobe could not be executed, or produced unusable output."""


def require_ffprobe() -> None:
    """Raise FFprobeError when the ffprobe binary is absent from PATH."""
    if shutil.which("ffprobe") is None:
        raise FFprobeError("ffprobe not found on PATH (ffmpeg installation required)")


def _exec(args: list[str], *, path: str, timeout_s: float) -> tuple[str, str]:
    """Run ffprobe with `args` (binary name excluded); return (stdout, stderr)."""
    require_ffprobe()
    cmd = ["ffprobe", *args, str(path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFprobeError(f"ffprobe timed out after {timeout_s}s on {path}", cause=exc) from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        raise FFprobeError(f"ffprobe failed for {path}: {exc}", cause=exc) from exc
    return proc.stdout, proc.stderr


def probe_text(
    path: str | Path,
    *,
    entries: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> str:
    """Query a single ffprobe `format=` entry and return its raw stdout scalar."""
    stdout, _ = _exec(
        ["-v", "error", "-show_entries", f"format={entries}", "-of", "default=nw=1:nk=1"],
        path=str(path),
        timeout_s=timeout_s,
    )
    return stdout.strip()


def probe_json(
    path: str | Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run the canonical `-show_format -show_streams -print_format json` query."""
    stdout, stderr = _exec(
        ["-v", "quiet", "-print_format", "json", "-show_format", "-show_streams"],
        path=str(path),
        timeout_s=timeout_s,
    )
    if not stdout.strip():
        raise FFprobeError(f"ffprobe produced no output for {path}: {stderr.strip()[:200]}")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise FFprobeError(f"ffprobe invalid JSON for {path}", cause=exc) from exc
    if not isinstance(data, dict):
        raise FFprobeError(f"ffprobe returned non-object JSON for {path}")
    return data


def parse_rational(value: Any) -> float | None:
    """Parse an ffprobe rational string ("30000/1001") or bare number into a float."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"n/a", "0/0", "none"}:
        return None
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return None


def coerce_float(value: Any) -> float | None:
    """Parse an ffprobe numeric field, tolerating "N/A" and empty strings."""
    parsed = parse_rational(value)
    return None if parsed is None else float(parsed)


def coerce_int(value: Any) -> int | None:
    """Parse an ffprobe integer field, tolerating "N/A" and empty strings."""
    parsed = parse_rational(value)
    return None if parsed is None else int(parsed)
