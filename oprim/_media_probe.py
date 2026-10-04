"""Media probe oprim — canonical metadata inspection for audio/video files.

Canonical authority for media inspection (SPEC P0 §4.1). `probe_media` is the
capability-oriented alias; `probe_duration` in `_probe_duration.py` is the
compatibility wrapper over this element.

Read-only (R0). Execution location is the caller's choice: pass `ffprobe_json`
to parse a pre-fetched payload, or `path` to run ffprobe locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from oprim._exceptions import OprimError, OprimNotFoundError
from oprim._ffprobe import coerce_float, coerce_int, parse_rational, probe_json


class MediaStream(BaseModel):
    type: str | None = None  # video / audio / subtitle
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    bitrate: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None


class MediaInfo(BaseModel):
    format_name: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    width: int | None = None  # first video stream
    height: int | None = None
    fps: float | None = None  # first video stream
    bitrate: int | None = None  # container-level bits/sec
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_sample_rate: int | None = None
    audio_channels: int | None = None
    is_video: bool = False
    is_audio: bool = False
    streams: list[MediaStream] = []


def _stream_fps(raw: dict[str, Any]) -> float | None:
    """Prefer avg_frame_rate, fall back to r_frame_rate (ffprobe reports both)."""
    for key in ("avg_frame_rate", "r_frame_rate"):
        parsed = parse_rational(raw.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _parse(data: dict[str, Any]) -> MediaInfo:
    fmt = data.get("format") or {}
    streams_raw = data.get("streams") or []
    streams: list[MediaStream] = []
    first_video: MediaStream | None = None
    first_audio: MediaStream | None = None
    for s in streams_raw:
        ct = s.get("codec_type")
        ms = MediaStream(
            type=ct,
            codec=s.get("codec_name"),
            width=coerce_int(s.get("width")),
            height=coerce_int(s.get("height")),
            fps=_stream_fps(s) if ct == "video" else None,
            duration_seconds=coerce_float(s.get("duration")),
            bitrate=coerce_int(s.get("bit_rate")),
            sample_rate=coerce_int(s.get("sample_rate")),
            channels=coerce_int(s.get("channels")),
            channel_layout=s.get("channel_layout"),
        )
        streams.append(ms)
        if ct == "video" and first_video is None:
            first_video = ms
        elif ct == "audio" and first_audio is None:
            first_audio = ms

    duration = coerce_float(fmt.get("duration"))
    if duration is None and streams:
        # Stream duration is the fallback when the container header lacks it
        # (common for raw streams and some webm files).
        known = [s.duration_seconds for s in streams if s.duration_seconds is not None]
        duration = max(known) if known else None

    return MediaInfo(
        format_name=fmt.get("format_name"),
        duration_seconds=duration,
        size_bytes=coerce_int(fmt.get("size")),
        width=first_video.width if first_video else None,
        height=first_video.height if first_video else None,
        fps=first_video.fps if first_video else None,
        bitrate=coerce_int(fmt.get("bit_rate")),
        video_codec=first_video.codec if first_video else None,
        audio_codec=first_audio.codec if first_audio else None,
        audio_sample_rate=first_audio.sample_rate if first_audio else None,
        audio_channels=first_audio.channels if first_audio else None,
        is_video=first_video is not None,
        is_audio=first_audio is not None,
        streams=streams,
    )


def media_probe(
    *,
    path: str | Path | None = None,
    ffprobe_json: str | None = None,
) -> MediaInfo:
    """Probe media file metadata (duration/fps/codecs/resolution/streams/bitrate).

    Read-only (R0). Canonical inspection atom: `probe_media` is an alias of this
    function, and `probe_duration` delegates here for its scalar result.

    Execution location is the caller's choice — pass `ffprobe_json` to parse a
    payload fetched elsewhere, or `path` to run ffprobe locally.

    Args:
        path: Media file path (required for local execution).
        ffprobe_json: Optional pre-fetched ffprobe `-print_format json` output.

    Returns:
        MediaInfo: container format, duration, size, bitrate, first video stream
        (codec/width/height/fps), first audio stream (codec/sample rate/channels),
        is_video/is_audio flags, and the full stream list.

    Raises:
        OprimNotFoundError: Neither path nor ffprobe_json given.
        OprimError: ffprobe missing, timed out, or returned invalid JSON.
    """
    if ffprobe_json is not None:
        try:
            data = json.loads(ffprobe_json)
        except json.JSONDecodeError as e:
            raise OprimError("ffprobe_json is not valid JSON", cause=e) from e
        if not isinstance(data, dict):
            raise OprimError("ffprobe_json is not a JSON object")
        return _parse(data)
    if not path:
        raise OprimNotFoundError("path or ffprobe_json is required")
    return _parse(probe_json(path))


def probe_media(
    *,
    path: str | Path | None = None,
    ffprobe_json: str | None = None,
) -> MediaInfo:
    """Capability-oriented alias for `media_probe` (SPEC §4.1 canonical name).

    Thin same-module alias: both names resolve to one implementation, so there
    is exactly one canonical media inspection element.
    """
    return media_probe(path=path, ffprobe_json=ffprobe_json)
