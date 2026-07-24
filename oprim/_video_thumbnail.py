"""Video thumbnail oprim — ffmpeg 抽帧生成缩略图(R0 只读)."""

from __future__ import annotations

import shutil
import subprocess
from typing import Literal

from pydantic import BaseModel

from oprim._exceptions import OprimError, OprimValidationError

# 输出编码 → ffmpeg 编码器/封装.
_FMT = {
    "jpeg": ("mjpeg", "image2pipe"),
    "png": ("png", "image2pipe"),
    "webp": ("libwebp", "image2pipe"),
}


class VideoThumbnail(BaseModel):
    data: bytes
    format: str


def _run_ffmpeg(args: list[str], timeout: int) -> bytes:
    if shutil.which("ffmpeg") is None:
        raise OprimError("ffmpeg not found (ffmpeg required)")
    try:
        proc = subprocess.run(
            ["ffmpeg", *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError("ffmpeg timed out", cause=e) from e
    if proc.returncode != 0 or not proc.stdout:
        err = proc.stderr.decode("utf-8", "replace")[-200:] if proc.stderr else ""
        raise OprimError(f"ffmpeg failed to extract frame: {err}")
    return proc.stdout


def video_thumbnail(
    *,
    path: str,
    at_seconds: float = 1.0,
    max_size: int = 320,
    fmt: Literal["jpeg", "png", "webp"] = "jpeg",
    timeout: int = 30,
) -> VideoThumbnail:
    """从视频指定时刻抽一帧并缩放为缩略图, 来自 `ffmpeg`.

    只读(R0, 不改源视频). ffmpeg 直接 seek+抽帧+等比缩放(scale ...
    force_original_aspect_ratio=decrease)输出到 stdout. 用于文件管理器视频缩略图.

    Args:
        path: 视频文件路径.
        at_seconds: 抽帧时刻(秒). 超过时长时 ffmpeg 取最后一帧.
        max_size: 缩略图最长边像素上限(> 0).
        fmt: 输出编码 jpeg / png / webp(webp 需 ffmpeg 编译含 libwebp).
        timeout: ffmpeg 超时秒.

    Returns:
        VideoThumbnail: data 为编码后帧字节, format 为 fmt.

    Raises:
        OprimValidationError: max_size <= 0 或 path 为空.
        OprimError: ffmpeg 缺失/超时/抽帧失败.
    """
    if not path:
        raise OprimValidationError("path is required")
    if max_size <= 0:
        raise OprimValidationError("max_size must be > 0")

    codec, container = _FMT[fmt]
    # -ss 放在 -i 前做快速 seek;scale 等比缩放到 max_size 内(-2 保证偶数边).
    scale = f"scale=w={max_size}:h={max_size}:force_original_aspect_ratio=decrease"
    args = [
        "-nostdin",
        "-v",
        "quiet",
        "-ss",
        str(max(0.0, at_seconds)),
        "-i",
        path,
        "-frames:v",
        "1",
        "-vf",
        scale,
        "-c:v",
        codec,
        "-f",
        container,
        "pipe:1",
    ]
    data = _run_ffmpeg(args, timeout)
    return VideoThumbnail(data=data, format=fmt)
