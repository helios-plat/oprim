"""oprim.media_content_parse — 媒体内容解析.

解析媒体文件基础信息（类型/大小/时长提示）；parser Protocol 注入时
委托深度解析（帧数/编码/缩略图等）。

Example:
    >>> r = await media_content_parse("/tmp/photo.jpg")
    >>> r["kind"]
    'image'
"""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError

KIND_BY_MIME: dict[str, str] = {
    "image": "image",
    "video": "video",
    "audio": "audio",
    "text": "text",
}


class MediaParseError(OprimError):
    """媒体解析失败。"""


@runtime_checkable
class MediaParser(Protocol):
    """深度解析协议（注入面）。"""

    async def parse(self, path: str, *, kind: str) -> dict[str, Any]: ...


async def media_content_parse(
    path: str | Path,
    *,
    kind: str | None = None,
    parser: MediaParser | None = None,
) -> dict[str, Any]:
    """解析媒体内容。

    Args:
        path: 媒体文件路径。
        kind: 强制类型（image/video/audio/text）；None 按 MIME 推断。
        parser: 深度解析器（注入，可选）。

    Returns:
        {"status": "ok", "path": str, "kind": str, "mime": str, "size_bytes": int,
         "meta": dict}

    Raises:
        MediaParseError: 文件缺失 / 类型不可判 / 深度解析失败。
        OprimValidationError: path 为空。
    """
    src = Path(path).expanduser()
    if not str(src):
        raise OprimValidationError("media_content_parse: path must not be empty")
    if not src.is_file():
        raise MediaParseError(f"media_content_parse: file not found: {src}")

    mime, _ = mimetypes.guess_type(str(src))
    mime = mime or "application/octet-stream"
    inferred = KIND_BY_MIME.get(mime.split("/")[0], "other")
    final_kind = kind or inferred
    if final_kind == "other":
        raise MediaParseError(
            f"media_content_parse: cannot determine kind for {src} (mime={mime})"
        )

    meta: dict[str, Any] = {"mime": mime}
    if parser is not None:
        try:
            deep = await parser.parse(str(src), kind=final_kind)
            if isinstance(deep, dict):
                meta.update(deep)
        except Exception as exc:
            raise MediaParseError(
                f"media_content_parse: parser failed: {exc}", cause=exc
            ) from exc

    return {
        "status": "ok",
        "path": str(src),
        "kind": final_kind,
        "mime": mime,
        "size_bytes": src.stat().st_size,
        "meta": meta,
    }
