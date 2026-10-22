"""oprim.media_publish_post — 媒体内容发布.

经注入的 publisher Protocol 把内容（含可选媒体附件）发布到目标渠道，
返回 post_id。

Example:
    >>> r = await media_publish_post("x", content="新视频来了", media_paths=["v.mp4"], publisher=p)
    >>> r["post_id"] != ""
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class MediaPublishError(OprimError):
    """媒体发布失败。"""


@runtime_checkable
class MediaPublisher(Protocol):
    """发布器协议（注入面）。"""

    async def publish(
        self, channel: str, *, content: str, media_paths: list[str]
    ) -> dict[str, Any]: ...


async def media_publish_post(
    channel: str,
    *,
    content: str,
    media_paths: list[str] | None = None,
    publisher: MediaPublisher,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """发布媒体帖子。

    Args:
        channel: 目标渠道（x/instagram/youtube/...）。
        content: 文案。
        media_paths: 媒体附件路径列表（可选）。
        publisher: 发布器（注入）。
        timeout: 超时秒数。

    Returns:
        {"status": "ok", "post_id": str, "channel": str, "media_count": int}

    Raises:
        MediaPublishError: 发布失败 / 附件缺失。
        OprimValidationError: channel / content 为空或 publisher 未注入。
    """
    if not channel or not channel.strip():
        raise OprimValidationError("media_publish_post: channel must not be empty")
    if not content.strip():
        raise OprimValidationError("media_publish_post: content must not be empty")
    if publisher is None:
        raise OprimValidationError("media_publish_post: publisher must be injected")

    media_paths = list(media_paths or [])
    for mp in media_paths:
        if not Path(mp).is_file():
            raise MediaPublishError(f"media_publish_post: media not found: {mp}")

    import asyncio

    try:
        result = await asyncio.wait_for(
            publisher.publish(channel, content=content, media_paths=media_paths),
            timeout=timeout,
        )
    except TimeoutError as exc:
        raise MediaPublishError(
            f"media_publish_post timed out after {timeout}s: {channel}", cause=exc
        ) from exc
    except Exception as exc:
        raise MediaPublishError(
            f"media_publish_post failed: {type(exc).__name__}: {exc}", cause=exc
        ) from exc

    post_id = ""
    if isinstance(result, dict):
        post_id = str(result.get("post_id", result.get("id", "")))
    return {
        "status": "ok",
        "post_id": post_id,
        "channel": channel,
        "media_count": len(media_paths),
        "response": result,
    }
