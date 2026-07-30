"""oprim.kling_v2_generate — 快手可灵 v2 master (fal) 高写实视频原语。

真人动作/一致性强,支持 negative_prompt 与朝向。与 ltx2_cloud_generate 并列作 video
原语;走 fal 队列(fal_queue_generate)。

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.kling_v2_generate import kling_v2_generate
    >>> out = asyncio.run(kling_v2_generate(
    ...     prompt="a dancer spinning, studio light", output_path=Path("clip.mp4"),
    ... ))

Raises:
    FalQueueError: FAL_API_KEY 缺失、fal 提交/轮询失败、超时或空产物。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._fal_queue_generate import _fal_aspect_ratio, fal_queue_generate

_ENDPOINT = "fal-ai/kling-video/v2/master/text-to-video"
# 默认负向提示:压制常见写实缺陷(崩手/多指/畸变)。
_DEFAULT_NEGATIVE = "blur, distort, low quality, deformed hands, extra fingers, bad anatomy"


async def kling_v2_generate(
    *,
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "9:16",
    duration_s: float = 5,
    negative_prompt: str = "",
    cfg_scale: float = 0.5,
    config: dict[str, Any] | None = None,
    **_kw: Any,
) -> Path:
    """生成一段可灵 v2 写实视频。

    Args:
        prompt: 文本提示。
        output_path: 产物落盘路径。
        aspect_ratio: {"9:16","16:9","1:1"};非法回退按 size 推导或 9:16。
        duration_s: 时长秒(下发为 "{int}")。
        negative_prompt: 负向提示;空则用内置写实缺陷压制默认值。
        cfg_scale: 提示遵循度。
        config: 覆盖 dict(FAL_API_KEY)。
        _kw: registry 注入的其余 kw 一律忽略(size 可参与 aspect_ratio 推导)。

    Returns:
        output_path。
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": f"{int(duration_s)}",
        "aspect_ratio": _fal_aspect_ratio(aspect_ratio, _kw),
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt or _DEFAULT_NEGATIVE,
    }
    return await fal_queue_generate(
        endpoint=_ENDPOINT, payload=payload, output_path=Path(output_path), config=config
    )
