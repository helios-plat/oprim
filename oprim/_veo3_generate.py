"""oprim.veo3_generate — Google Veo 3 (fal-ai/veo3/fast) 高写实视频原语。

oprim 内置的 ltx2_cloud(fal-ai/ltx-video 基础版)写实/人体解剖弱(手崩、768x512
低清)。Veo 3 写实与解剖最佳,支持原生音频与 negative_prompt,适合"真人一样"的需求。
与 ltx2_cloud_generate / vibevoice_synthesize 并列作 video 原语;走 fal 队列
(fal_queue_generate)。

Example:
    >>> import asyncio
    >>> from pathlib import Path
    >>> from oprim.veo3_generate import veo3_generate
    >>> out = asyncio.run(veo3_generate(
    ...     prompt="a chef plating a dish, cinematic", output_path=Path("clip.mp4"),
    ...     aspect_ratio="9:16", duration_s=8,
    ... ))

Raises:
    FalQueueError: FAL_API_KEY 缺失、fal 提交/轮询失败、超时或空产物。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._fal_queue_generate import _fal_aspect_ratio, fal_queue_generate

_ENDPOINT = "fal-ai/veo3/fast"


async def veo3_generate(
    *,
    prompt: str,
    output_path: Path,
    aspect_ratio: str = "9:16",
    duration_s: float = 8,
    negative_prompt: str = "",
    generate_audio: bool = True,
    resolution: str = "720p",
    config: dict[str, Any] | None = None,
    **_kw: Any,
) -> Path:
    """生成一段 Veo 3 竖屏(或指定朝向)写实视频。

    Args:
        prompt: 文本提示。
        output_path: 产物落盘路径。
        aspect_ratio: {"9:16","16:9","1:1"};非法值回退按 size 推导或 9:16。
        duration_s: 时长秒(下发为 "{int}s")。
        negative_prompt: 负向提示(非空才下发)。
        generate_audio: 是否请求原生音频。
        resolution: 分辨率档,如 "720p"。
        config: 覆盖 dict(FAL_API_KEY)。
        _kw: registry 注入的其余 kw(如 size/mode/reference_image)一律忽略;
            仅 size 可参与 aspect_ratio 推导。

    Returns:
        output_path。
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": _fal_aspect_ratio(aspect_ratio, _kw),
        "duration": f"{int(duration_s)}s",
        "generate_audio": generate_audio,
        "resolution": resolution,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    return await fal_queue_generate(
        endpoint=_ENDPOINT, payload=payload, output_path=Path(output_path), config=config
    )
