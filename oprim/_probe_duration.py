"""oprim.probe_duration — ffprobe 实测媒体文件时长(秒)。

Backward-compatibility shim for the HEVI 3O §2 consumer contract. The canonical
implementation lives in :mod:`oprim._media_probe`; this module only adapts it to
the historical ``probe_duration(path) -> float`` signature and error type. No
subprocess / ffprobe logic is duplicated here.
"""

from __future__ import annotations

import math
from pathlib import Path

from oprim._media_probe import media_probe


class ProbeDurationError(Exception):
    """ffprobe duration probing failed."""


def probe_duration(path: str | Path) -> float:
    """返回媒体文件真实时长(秒);探测失败抛 :class:`ProbeDurationError`.

    Thin compatibility wrapper over the canonical
    :func:`oprim._media_probe.media_probe`: it accepts ``str`` or
    :class:`pathlib.Path`, delegates probing to ``media_probe(path=str(path))``
    and returns the parsed ``MediaInfo.duration_seconds``.

    Args:
        path: 媒体文件路径(``str`` 或 :class:`pathlib.Path`).

    Returns:
        float: 媒体时长(秒),必须为有限正数.

    Raises:
        ProbeDurationError: ``media_probe`` 失败,或时长缺失/无法解析/非法.
    """
    try:
        info = media_probe(path=str(path))
    except Exception as exc:  # noqa: BLE001 - 统一收敛为消费契约异常
        raise ProbeDurationError(f"ffprobe failed for {path}: {exc}") from exc

    duration = info.duration_seconds
    if duration is None:
        raise ProbeDurationError(f"missing ffprobe duration for {path}")

    try:
        seconds = float(duration)
    except (TypeError, ValueError) as exc:
        raise ProbeDurationError(
            f"unparsable ffprobe duration for {path}: {duration!r}"
        ) from exc

    if not math.isfinite(seconds) or seconds <= 0:
        raise ProbeDurationError(f"invalid ffprobe duration for {path}: {duration!r}")

    return seconds
