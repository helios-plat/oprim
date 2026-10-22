"""oprim.adaptive_node_extract — 单次通过自适应语义与正则寻找目标网页节点.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: re (stdlib).

例:
    >>> html = "<div class='article-content'><p>Hello</p></div>"
    >>> result = adaptive_node_extract(html, target_semantic="article-content")
    >>> result["status"]
    'success'
"""

from __future__ import annotations

import re
from typing import Any


def adaptive_node_extract(
    html_content: str,
    *,
    target_semantic: str = "article-content",
) -> dict[str, Any]:
    """单次通过自适应语义与正则寻找目标网页节点 (Scrapling 机制).

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    算法:
    1. 按 class/id 语义正则匹配目标节点
    2. 回退到 <main>/<article> HTML5 语义标签
    3. 最终回退：取前 1000 字符文本
    4. 清理 HTML 标签，保留纯文本

    Args:
        html_content: HTML 文本
        target_semantic: 目标节点语义（如 "article-content", "main-body"）

    Returns:
        {
            "status": "success" | "fallback",
            "extracted_content": str,
            "anchor_confidence": float (0.0-1.0),
        }
    """
    if not html_content:
        return {
            "status": "fallback",
            "extracted_content": "",
            "anchor_confidence": 0.0,
        }

    # 策略 1: 按 target_semantic 匹配 class/id
    pattern = (
        r"<(?:div|section|article|main)[^>]*"
        r"(?:class|id)\s*=\s*['\"]([^'\"]*"
        + re.escape(target_semantic)
        + r"[^'\"]*)['\"][^>]*>(.*?)</(?:div|section|article|main)>"
    )
    match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)

    if match:
        extracted = _strip_html(match.group(2).strip())
        return {
            "status": "success",
            "extracted_content": extracted,
            "anchor_confidence": 0.95,
        }

    # 策略 2: 回退到 HTML5 语义标签
    for tag in ("article", "main"):
        fallback_pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        fmatch = re.search(fallback_pattern, html_content, re.DOTALL | re.IGNORECASE)
        if fmatch:
            extracted = _strip_html(fmatch.group(1).strip())
            return {
                "status": "success",
                "extracted_content": extracted,
                "anchor_confidence": 0.70,
            }

    # 策略 3: 最终回退 — 清理所有标签取前 2000 字符
    cleaned = _strip_html(html_content)
    return {
        "status": "fallback",
        "extracted_content": cleaned[:2000],
        "anchor_confidence": 0.30,
    }


def _strip_html(text: str) -> str:
    """移除 HTML 标签，合并空白."""
    text = re.sub(r"<[^<]+?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
