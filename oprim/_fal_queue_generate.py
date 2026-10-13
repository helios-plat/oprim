"""oprim.fal_queue_generate — fal.ai 队列制模型的通用提交/轮询/下载工具。

fal 的高写实视频模型(veo3 / kling_v2 / hailuo,以及后续的 lipsync)都是队列制:
``POST queue.fal.run/{endpoint}`` → 轮询 status_url 至 COMPLETED → 取 response_url →
下载产物。这些原语共用本工具。轮询带 **总超时 deadline**,避免 fal 队列卡住时无限挂。

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.fal_queue_generate import fal_queue_generate
    >>> out = asyncio.run(fal_queue_generate(
    ...     endpoint="fal-ai/veo3/fast",
    ...     payload={"prompt": "a cat on the moon", "aspect_ratio": "9:16"},
    ...     output_path=Path("clip.mp4"),
    ... ))

Raises:
    FalQueueError: FAL_API_KEY 缺失、提交/轮询失败、超时,或产物为空。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from oprim._config import cfg

logger = logging.getLogger(__name__)


class FalQueueError(Exception):
    """fal.ai queue submission / polling / download failed."""


_FAL_BASE = "https://queue.fal.run"
_POLL_INTERVAL_S = 5.0
_DEFAULT_TIMEOUT_S = 600.0


def _fal_aspect_ratio(aspect_ratio: str | None, kw: dict[str, Any]) -> str:
    """解析朝向:优先显式合法值,否则从 (w, h) size 推导;缺省 9:16(短视频主场景)。"""
    if aspect_ratio in ("9:16", "16:9", "1:1"):
        return aspect_ratio  # type: ignore[return-value]
    size = kw.get("size")
    if isinstance(size, (tuple, list)) and len(size) == 2:
        w, h = size
        return "16:9" if w > h else "1:1" if w == h else "9:16"
    return "9:16"


async def fal_queue_generate(
    *,
    endpoint: str,
    payload: dict[str, Any],
    output_path: Path,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    config: dict[str, Any] | None = None,
) -> Path:
    """提交 fal 队列任务、轮询至完成、下载视频到 output_path。

    Args:
        endpoint: fal 端点,如 "fal-ai/veo3/fast"。
        payload: 提交给该端点的 JSON body。
        output_path: 产物落盘路径。
        timeout_s: 轮询总超时(秒);超过即抛 FalQueueError(治 fal 队列无限挂)。
        config: 覆盖 dict(FAL_API_KEY),缺省回退 env/cfg。

    Returns:
        output_path。

    Raises:
        FalQueueError: 缺 key / 提交失败 / 任务失败 / 超时 / 空产物。
    """
    cfg_dict = config or {}
    api_key: str = cfg_dict.get("FAL_API_KEY") or cfg.get("FAL_API_KEY", "")  # type: ignore[assignment]
    if not api_key:
        raise FalQueueError("FAL_API_KEY not configured")

    headers = {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import httpx

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as c:
        sub = await c.post(f"{_FAL_BASE}/{endpoint}", json=payload, headers=headers)
        if sub.status_code not in (200, 201, 202):
            raise FalQueueError(f"fal submit {sub.status_code}: {sub.text[:300]}")
        data = sub.json()
        status_url = data.get("status_url")
        response_url = data.get("response_url")

        # 队列制:轮询至完成(带 deadline)。同步返回(无 status_url)则 data 已是结果。
        while status_url:
            if loop.time() > deadline:
                raise FalQueueError(f"fal job timeout after {timeout_s:.0f}s ({endpoint})")
            st = (await c.get(status_url, headers=headers)).json()
            status = st.get("status", "")
            if status == "COMPLETED":
                data = (await c.get(response_url, headers=headers)).json()
                break
            if status in ("FAILED", "CANCELLED", "ERROR"):
                raise FalQueueError(f"fal job {status} ({endpoint}): {str(st)[:300]}")
            await asyncio.sleep(_POLL_INTERVAL_S)

        video_url = (
            (data.get("video") or {}).get("url")
            or data.get("video_url")
            or (data.get("output") or {}).get("video_url")
        )
        if not video_url:
            raise FalQueueError(f"fal: no video url in response ({endpoint}): {str(data)[:300]}")
        dl = await c.get(video_url, timeout=httpx.Timeout(300.0))
        if dl.status_code != 200:
            raise FalQueueError(f"fal video download {dl.status_code}")
        output_path.write_bytes(dl.content)

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise FalQueueError(f"fal produced no/empty output ({endpoint})")
    logger.info("fal %s → %s (%d bytes)", endpoint, output_path.name, output_path.stat().st_size)
    return output_path
