"""oprim._llm_router — LLM 智能路由原语 (RouteLLM 3O 内化)。

任务特征提取 → 档位判定 → 路由矩阵查表 → 决策 (provider/model)。
纯规则、确定性、零外部依赖; 矩阵 JSON 可配置热重载 (mtime)。

分层: oprim (原语) — 只做决策, 不做网络调用。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# 默认路由矩阵 (veya1.1 别名 → 档位 → provider/model)
DEFAULT_MATRIX: dict[str, Any] = {
    "alias": "veya1.1",
    "routes": {
        "quick": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "text": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "tool": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "code": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "reason": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "long": {"provider": "deepseek", "model": "deepseek-v4-flash"},
        "vision": {"provider": "dashscope", "model": "qwen3.7-flash"},
    },
    "fallback": {"provider": "deepseek", "model": "deepseek-v4-flash"},
    "thresholds": {"long_tokens": 6000, "quick_tokens": 300},
    "parallelism": 4,
}

_ROUTER_FILE = Path.home() / ".veya" / "llm-router.json"

# 档位特征 (关键词 → 档位)
_CODE_HINTS = re.compile(r"```|\b(def|class|function|import|const|return)\b", re.IGNORECASE)
_REASON_HINTS = re.compile(
    r"\b(证明|推导|为什么|原因|证明题|数学|calculate|prove|derive)\b",  # noqa: E501
    re.IGNORECASE)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """粗略 token 估算 (chars/4, 中文偏保守用 /3 混合: 简单按 1 token ≈ 3.5 chars)。"""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total += len(str(block.get("text", "")))
    return int(total / 3.5)


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("image_url", "image"):
                    return True
        if isinstance(content, str) and re.search(
                r"data:image/|!\[.*\]\(.*\.(png|jpg|jpeg)", content):  # noqa: E501
            return True
    return False


def _classify(messages: list[dict[str, Any]], tools: list | None,
              thresholds: dict[str, int]) -> str:
    """任务类型分类: vision > reason > code > tool > long > quick/text。"""
    if _has_image(messages):
        return "vision"
    text = " ".join(
        str(m.get("content", "")) for m in messages
        if isinstance(m.get("content"), str))
    if _REASON_HINTS.search(text):
        return "reason"
    if _CODE_HINTS.search(text):
        return "code"
    if tools:
        return "tool"
    tokens = _estimate_tokens(messages)
    if tokens > thresholds.get("long_tokens", 6000):
        return "long"
    if tokens <= thresholds.get("quick_tokens", 300):
        return "quick"
    return "text"


# ── 矩阵加载 (热重载) ────────────────────────────────────────────────

_cache: dict[str, Any] = {"mtime": 0.0, "matrix": DEFAULT_MATRIX}


def load_matrix(path: str = "") -> dict[str, Any]:
    """加载路由矩阵 (~/.veya/llm-router.json), mtime 热重载; 失败回退默认。"""
    p = Path(path or _ROUTER_FILE)
    try:
        mtime = p.stat().st_mtime
        if mtime != _cache["mtime"]:
            data = json.loads(p.read_text(encoding="utf-8"))
            matrix = DEFAULT_MATRIX | data
            matrix["routes"] = {**DEFAULT_MATRIX["routes"], **(data.get("routes") or {})}
            _cache["matrix"] = matrix
            _cache["mtime"] = mtime
    except (OSError, json.JSONDecodeError):
        pass  # 无文件/损坏 → 默认矩阵
    return _cache["matrix"]


def route_decision(
    messages: list[dict[str, Any]],
    tools: list | None = None,
    matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """主入口: 特征 → 档位 → 决策 {route, provider, model, reason}。

    永不抛异常: 矩阵损坏/未知档 → fallback。
    """
    m = matrix or load_matrix()
    routes = m.get("routes") or {}
    fallback = m.get("fallback") or DEFAULT_MATRIX["fallback"]
    thresholds = {**DEFAULT_MATRIX["thresholds"], **(m.get("thresholds") or {})}

    route = _classify(messages, tools, thresholds)
    target = routes.get(route) or fallback
    return {
        "route": route,
        "provider": target.get("provider", fallback["provider"]),
        "model": target.get("model", fallback["model"]),
        "reason": f"route={route}",
        "tokens": _estimate_tokens(messages),
    }


__all__ = ["DEFAULT_MATRIX", "load_matrix", "route_decision",
           "_classify", "_estimate_tokens", "_has_image"]
