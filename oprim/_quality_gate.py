"""oprim._quality_gate — 路由响应质量闸门原语 (分层路由 v2)。

响应快速校验 (纯规则, 零 LLM 开销): 空响应/错误标记/异常短 → 不通过。
闸门失败 → 装配层升级档位重试 (flash → frontier, 最多 1 次)。
"""

from __future__ import annotations

import re
from typing import Any

# 错误/降级标记 (响应中出现 → 视为低质量)
_ERROR_MARKERS = re.compile(
    r"(?i)(shim response|not configured|error|exception|traceback|"
    r"timeout|超时|失败|不可用|internal server error)")
_EMPTYISH = re.compile(r"^\\s*$")


def extract_content(result: dict[str, Any]) -> str:
    """从 llm_call 结果提取文本 (choices 结构兼容)。"""
    try:
        content = result["choices"][0]["message"].get("content", "")
        if isinstance(content, list):  # Anthropic blocks
            parts = [b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"]
            return "".join(parts)
        return str(content or "")
    except (KeyError, IndexError, TypeError):
        return str(result.get("output", "") or "")


def quality_check(result: dict[str, Any]) -> dict[str, Any]:
    """质量闸门: 返回 {ok, reason}。

    通过条件: 非空 + 无错误标记 + 长度 ≥ 3。
    """
    content = extract_content(result)
    if _EMPTYISH.match(content):
        return {"ok": False, "reason": "empty_response"}
    if _ERROR_MARKERS.search(content):
        return {"ok": False, "reason": "error_marker"}
    if len(content.strip()) < 3:
        return {"ok": False, "reason": "too_short"}
    return {"ok": True, "reason": "pass"}


__all__ = ["quality_check", "extract_content"]
