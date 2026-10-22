"""oprim.segment_media — physical/boundary segmentation of a media file.

Single atomic detection pass: returns contiguous `[start, end)` spans derived
from objective signal boundaries (scene change via ffmpeg ``scdet``, silence
via ``silencedetect``, or caller-supplied hints). No semantics, no naming, no
story — classifying segments is oskill's job (see `segment_video_semantically`).

Raises:
    SegmentMediaError: Validation failure or FFmpeg error.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from obase.ffmpeg import FFmpegError
from obase.ffmpeg import run as ffmpeg_run
from pydantic import BaseModel

_SCD_SCORE_RE = re.compile(r"lavfi\.scd\.score:\s*([0-9.eE+-]+)")
_SCD_TIME_RE = re.compile(r"lavfi\.scd\.time:\s*([0-9.eE+-]+)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[0-9.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[0-9.]+)")


class SegmentMediaError(Exception):
    """Segmentation failed."""


class MediaSegment(BaseModel):
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class MediaSegmentList(BaseModel):
    source: Path
    method: str
    total_duration: float | None = None
    segments: list[MediaSegment] = []


def _spans_from_boundaries(
    boundaries: list[float], *, total_duration: float | None, min_duration: float
) -> list[MediaSegment]:
    """Turn cut points into contiguous spans, dropping degenerate slivers."""
    if total_duration is not None and total_duration <= 0:
        return []
    edges = [0.0, *boundaries]
    if total_duration is not None:
        edges.append(total_duration)
    spans: list[MediaSegment] = []
    for start, end in zip(edges, edges[1:], strict=False):
        if end - start < min_duration:
            continue
        spans.append(MediaSegment(index=len(spans), start=round(start, 6), end=round(end, 6)))
    return spans


async def segment_media(
    *,
    media_path: Path,
    method: Literal["scene", "silence", "hints"] = "scene",
    boundary_hints: list[float] | None = None,
    threshold: float = 10.0,
    min_duration: float = 0.5,
    total_duration: float | None = None,
    timeout_s: float = 300.0,
) -> MediaSegmentList:
    """Detect physical segment boundaries in a media file.

    Args:
        media_path: Source audio or video file.
        method: Detection strategy. `"scene"` uses ffmpeg `scdet` (video only),
            `"silence"` uses `silencedetect` (audio), `"hints"` treats
            `boundary_hints` as the cut points.
        boundary_hints: Cut points in seconds. Required for `method="hints"`.
        threshold: Detector sensitivity, on ffmpeg's native 0–100 `scdet` scale.
            Higher means fewer cuts. Only used when `method="scene"`.
        min_duration: Segments shorter than this are dropped.
        total_duration: Source duration. When given, the trailing span up to the
            end is emitted as a final segment; otherwise detection stops at the
            last detected boundary.
        timeout_s: FFmpeg timeout in seconds.

    Returns:
        MediaSegmentList of contiguous spans. Empty when nothing was detected.

    Raises:
        SegmentMediaError: Invalid arguments or FFmpeg failure.
    """
    if not media_path.exists():
        raise SegmentMediaError(f"Media file not found: {media_path}")
    if not 0 < threshold <= 100:
        raise SegmentMediaError("threshold must be in (0, 100] (ffmpeg scdet scale)")
    if min_duration <= 0:
        raise SegmentMediaError("min_duration must be > 0")
    if total_duration is not None and total_duration <= 0:
        raise SegmentMediaError("total_duration must be > 0 when provided")

    if method == "hints":
        if boundary_hints is None:
            raise SegmentMediaError("method='hints' requires boundary_hints")
        cuts = sorted({round(float(h), 6) for h in boundary_hints})
        return MediaSegmentList(
            source=media_path,
            method=method,
            total_duration=total_duration,
            segments=_spans_from_boundaries(
                cuts, total_duration=total_duration, min_duration=min_duration
            ),
        )

    if method == "scene":
        stderr = await _run_detector(
            media_path, ["-vf", f"scdet=threshold={threshold}"], timeout_s=timeout_s
        )
        cuts = _scdet_cuts(stderr)
    elif method == "silence":
        stderr = await _run_detector(
            media_path, ["-af", f"silencedetect=n=-40dB:d={min_duration}"], timeout_s=timeout_s
        )
        cuts = _silence_cuts(stderr)
    else:
        raise SegmentMediaError(f"Unknown method: {method!r}")

    return MediaSegmentList(
        source=media_path,
        method=method,
        total_duration=total_duration,
        segments=_spans_from_boundaries(
            sorted(set(cuts)), total_duration=total_duration, min_duration=min_duration
        ),
    )


def _scdet_cuts(stderr: str) -> list[float]:
    """Pair up ffmpeg's scdet score/time readings into cut points.

    ffmpeg emits `lavfi.scd.score: <0-100>, lavfi.scd.time: <seconds>` per event.
    Scores and times are read independently so a version that puts them on
    separate lines still parses; the shorter list wins.
    """
    times = [float(m.group(1)) for m in _SCD_TIME_RE.finditer(stderr)]
    scores = [float(m.group(1)) for m in _SCD_SCORE_RE.finditer(stderr)]
    del scores  # scores gate the emission; the filter already applied `threshold`
    return times


def _silence_cuts(stderr: str) -> list[float]:
    """Prefer silence_end cut points; fall back to negated silence_start."""
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(stderr)]
    if ends:
        return ends
    return [-float(m.group(1)) for m in _SILENCE_START_RE.finditer(stderr)]


async def _run_detector(media_path: Path, filter_args: list[str], *, timeout_s: float) -> str:
    """Run ffmpeg as a null-output analysis pass and return its stderr log."""
    args = ["-i", str(media_path), *filter_args, "-f", "null", "-"]
    try:
        return await ffmpeg_run(args=args, timeout_s=timeout_s)
    except FFmpegError as exc:
        raise SegmentMediaError(f"FFmpeg segmentation failed: {exc}") from exc
