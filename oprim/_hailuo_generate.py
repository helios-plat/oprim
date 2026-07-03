"""oprim.hailuo_generate — MiniMax 海螺 02 standard (fal) 高写实视频原语。

写实、性价比高。与 ltx2_cloud_generate 并列作 video 原语;走 fal 队列
(fal_queue_generate)。海螺 standard 端点为纯 t2v,不吃 aspect_ratio(由 prompt_optimizer
优化);朝向如需可后续换 pro 档。

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.hailuo_generate import hailuo_generate
    >>> out = asyncio.run(hailuo_generate(
    ...     prompt="a cat walking on a fence", output_path=Path("clip.mp4"),
    ... ))

Raises:
    FalQueueError: FAL_API_KEY 缺失、fal 提交/轮询失败、超时或空产物。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._fal_queue_generate import fal_queue_generate

_ENDPOINT = "fal-ai/minimax/hailuo-02/standard/text-to-video"


async def hailuo_generate(
    *,
    prompt: str,
    output_path: Path,
    duration_s: float = 6,
    resolution: str = "768P",
    config: dict[str, Any] | None = None,
    **_kw: Any,
) -> Path:
    """生成一段海螺 02 写实视频。

    Args:
        prompt: 文本提示。
        output_path: 产物落盘路径。
        duration_s: 时长秒(下发为 "{int}")。
        resolution: 分辨率档,如 "768P"。
        config: 覆盖 dict(FAL_API_KEY)。
        _kw: registry 注入的其余 kw 一律忽略。

    Returns:
        output_path。
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": f"{int(duration_s)}",
        "resolution": resolution,
        "prompt_optimizer": True,
    }
    return await fal_queue_generate(
        endpoint=_ENDPOINT, payload=payload, output_path=Path(output_path), config=config
    )
