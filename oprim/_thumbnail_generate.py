"""Image thumbnail oprim — 从图像字节生成缩略图(R0 纯计算)."""

from __future__ import annotations

import io
from typing import Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from oprim._exceptions import OprimValidationError

_FMT_PIL = {"webp": "WEBP", "jpeg": "JPEG", "png": "PNG"}


class ThumbnailResult(BaseModel):
    data: bytes  # 缩略图二进制
    format: str  # webp / jpeg / png
    width: int
    height: int
    source_width: int
    source_height: int


def thumbnail_generate(
    *,
    image_bytes: bytes,
    max_size: int = 256,
    fmt: Literal["webp", "jpeg", "png"] = "webp",
    quality: int = 80,
) -> ThumbnailResult:
    """把图像字节缩成不超过 max_size 的缩略图, 返回编码后的字节.

    纯计算(R0). 保持宽高比(最长边 = max_size), 自动按 EXIF 方向校正, 对
    JPEG/WEBP 统一转 RGB(丢 alpha). 用于文件管理器缩略图网格.

    Args:
        image_bytes: 原始图像字节(任意 Pillow 可解码格式).
        max_size: 缩略图最长边像素上限(> 0).
        fmt: 输出编码 webp / jpeg / png.
        quality: 有损编码质量 1-100(png 忽略).

    Returns:
        ThumbnailResult: data 为编码后字节, width/height 为缩略图尺寸,
            source_* 为原图尺寸.

    Raises:
        OprimValidationError: max_size <= 0、image_bytes 空、或不是可解码图像.
    """
    if max_size <= 0:
        raise OprimValidationError("max_size must be > 0")
    if not image_bytes:
        raise OprimValidationError("image_bytes is empty")

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise OprimValidationError(f"not a decodable image: {e}") from e

    src_w, src_h = img.size
    # 尊重 EXIF 方向(手机竖拍),再缩放.
    img = ImageOps.exif_transpose(img)

    pil_fmt = _FMT_PIL[fmt]
    if pil_fmt in ("JPEG", "WEBP") and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img.thumbnail((max_size, max_size), Image.LANCZOS)

    buf = io.BytesIO()
    save_kwargs: dict[str, object] = {}
    if pil_fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = max(1, min(100, quality))
    img.save(buf, format=pil_fmt, **save_kwargs)

    return ThumbnailResult(
        data=buf.getvalue(),
        format=fmt,
        width=img.width,
        height=img.height,
        source_width=src_w,
        source_height=src_h,
    )
