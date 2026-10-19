"""oprim.probe_duration — compatibility wrapper returning media duration in seconds.

Superseded by `media_probe` (SPEC §8.1): duration is one field of the canonical
`MediaInfo`. Kept as a thin wrapper so existing callers keep working — the
underlying implementation is the single `_ffprobe` executor, never a second
ffprobe invocation path.
"""

from __future__ import annotations

from pathlib import Path

from oprim._ffprobe import FFprobeError, probe_text


class ProbeDurationError(Exception):
    """ffprobe duration probing failed."""


def probe_duration(path: Path | str) -> float:
    """Return the media duration in seconds.

    Compatibility wrapper: prefer `media_probe(path=...).duration_seconds`, which
    returns the full metadata set in one ffprobe call.

    Args:
        path: Media file path.

    Returns:
        Duration in seconds.

    Raises:
        ProbeDurationError: ffprobe failed, or returned an unparsable value.
    """
    try:
        out = probe_text(path, entries="duration")
    except FFprobeError as exc:
        raise ProbeDurationError(f"ffprobe failed for {path}: {exc}") from exc
    try:
        return float(out)
    except ValueError as exc:
        raise ProbeDurationError(f"unparsable ffprobe duration for {path}: {out!r}") from exc
