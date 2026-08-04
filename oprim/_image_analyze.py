"""oprim.image_analyze — 单次多模态图像/文档结构化分析.

薄组合 oprim.image_understand（VLM 调用，provider 经 obase.ProviderRegistry
解析），输出标准化 dict。

Example:
    >>> r = await image_analyze("photo.png", "描述这张图", provider="qwen_vl")
    >>> r["text"].strip() != ""
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oprim._exceptions import OprimError, OprimValidationError


class ImageAnalyzeError(OprimError):
    """图像分析失败。"""


async def image_analyze(
    image_path: str | Path,
    *,
    prompt: str,
    provider: str,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """对图像/文档执行单次多模态分析。

    Args:
        image_path: 图像文件路径。
        prompt: 分析指令。
        provider: VLM provider 名（obase.ProviderRegistry vlm 类别）。
        timeout_s: 超时秒数。

    Returns:
        {"status": "ok", "text": str, "provider": str, "image_path": str}

    Raises:
        ImageAnalyzeError: VLM 调用失败。
        OprimValidationError: 参数缺失。
    """
    if not prompt.strip():
        raise OprimValidationError("image_analyze: prompt must not be empty")
    if not provider:
        raise OprimValidationError("image_analyze: provider must not be empty")

    # 惰性 import：image_understand 依赖 obase（可选），无 obase 环境不炸导入
    from oprim.image_understand import ImageUnderstandError, image_understand

    try:
        text = await image_understand(
            provider=provider,
            image_path=Path(image_path),
            prompt=prompt,
            timeout_s=timeout_s,
        )
    except ImageUnderstandError as exc:
        raise ImageAnalyzeError(f"image_analyze failed: {exc}", cause=exc) from exc

    return {
        "status": "ok",
        "text": text,
        "provider": provider,
        "image_path": str(image_path),
    }
