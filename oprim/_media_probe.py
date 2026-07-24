"""Media probe oprim — ffprobe 提取音视频元数据(只读 R0)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from pydantic import BaseModel

from oprim._exceptions import OprimError, OprimNotFoundError


class MediaStream(BaseModel):
    type: str | None = None  # video / audio / subtitle
    codec: str | None = None
    width: int | None = None
    height: int | None = None


class MediaInfo(BaseModel):
    format_name: str | None = None
    duration_seconds: float | None = None
    size_bytes: int | None = None
    width: int | None = None  # first video stream
    height: int | None = None
    is_video: bool = False
    is_audio: bool = False
    streams: list[MediaStream] = []


def _run_ffprobe(path: str) -> dict[str, Any]:
    """跑 `ffprobe -show_format -show_streams -print_format json <path>`. 供测试 monkeypatch."""
    if shutil.which("ffprobe") is None:
        raise OprimError("ffprobe not found (ffmpeg required)")
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError(f"ffprobe timed out on {path}", cause=e) from e
    if not proc.stdout.strip():
        raise OprimError(f"ffprobe produced no output for {path}: {proc.stderr.strip()[:200]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise OprimError(f"ffprobe invalid JSON for {path}", cause=e) from e


def _parse(data: dict[str, Any]) -> MediaInfo:
    fmt = data.get("format") or {}
    streams_raw = data.get("streams") or []
    streams: list[MediaStream] = []
    first_video: MediaStream | None = None
    is_video = is_audio = False
    for s in streams_raw:
        ct = s.get("codec_type")
        ms = MediaStream(
            type=ct,
            codec=s.get("codec_name"),
            width=s.get("width"),
            height=s.get("height"),
        )
        streams.append(ms)
        if ct == "video":
            is_video = True
            if first_video is None:
                first_video = ms
        elif ct == "audio":
            is_audio = True

    dur = fmt.get("duration")
    size = fmt.get("size")
    return MediaInfo(
        format_name=fmt.get("format_name"),
        duration_seconds=float(dur) if dur is not None else None,
        size_bytes=int(size) if size is not None else None,
        width=first_video.width if first_video else None,
        height=first_video.height if first_video else None,
        is_video=is_video,
        is_audio=is_audio,
        streams=streams,
    )


def media_probe(
    *,
    path: str | None = None,
    ffprobe_json: str | None = None,
) -> MediaInfo:
    """探测媒体文件元数据(时长/编解码/分辨率/流),来自 `ffprobe`.

    只读(R0). 用于文件管理器媒体预览、给视频/音频文件标注信息.

    执行位置由调用方决定:传 ffprobe_json 只解析(调用方在别处取输出);否则
    需 path 且本地跑 ffprobe.

    Args:
        path: 媒体文件路径(本地执行时必需).
        ffprobe_json: 可选. 预取的 ffprobe `-print_format json` 原始输出.

    Returns:
        MediaInfo: format/duration/size/首个视频流分辨率/is_video/is_audio/streams.

    Raises:
        OprimNotFoundError: 既未给 path 也未给 ffprobe_json.
        OprimError: ffprobe 缺失、超时或输出非法.
    """
    if ffprobe_json is not None:
        try:
            data = json.loads(ffprobe_json)
        except json.JSONDecodeError as e:
            raise OprimError("ffprobe_json is not valid JSON", cause=e) from e
        return _parse(data)
    if not path:
        raise OprimNotFoundError("path or ffprobe_json is required")
    return _parse(_run_ffprobe(path))
