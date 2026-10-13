"""oprim._parallel_llm — 长输入并行分派原语 (快速回答)。

长 prompt → 切分 (段落/句子边界) → 并行调用 (调用方注入 caller) → 聚合。
纯编排原语: 不接触网络, caller 由上层注入 (3O 零反向依赖)。

失败段隔离: 单段失败/超时 → 标记, 不阻塞聚合。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_SENTENCE_RE = re.compile(r"(?<=[。！？.!?])\s+")


def split_prompt(prompt: str, max_chunks: int = 4, min_len: int = 200) -> list[str]:
    """长文切分: 优先段落, 不足则句子聚合; 返回 ≤ max_chunks 块。

    块数 = min(max_chunks, 内容规模); 少于 min_len 不切。
    """
    if len(prompt) <= min_len * 2:
        return [prompt]
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(prompt) if p.strip()]

    chunks: list[str] = []
    if len(paragraphs) > 1:
        # 段落数 ≤ max_chunks → 直接; 否则合并段落到块
        if len(paragraphs) <= max_chunks:
            return paragraphs
        per = max(1, len(paragraphs) // max_chunks)
        for i in range(0, len(paragraphs), per):
            chunk = "\n\n".join(paragraphs[i:i + per])
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break
        # 兜底: 剩余段落并入最后一块
        if len(chunks) < max_chunks and chunks:
            chunks[-1] += "\n\n" + "\n\n".join(paragraphs[len(chunks) * per:])
    else:
        # 无段落 → 按句子聚合
        sentences = [s for s in _SENTENCE_RE.split(prompt) if s.strip()]
        per = max(1, len(sentences) // max_chunks)
        for i in range(0, len(sentences), per):
            chunks.append("".join(sentences[i:i + per]))
            if len(chunks) >= max_chunks:
                break
    return chunks or [prompt]


def aggregate_results(results: list[dict[str, Any]], *, title: str = "") -> str:
    """并行结果聚合: 按块序拼接 + 失败段标记。"""
    parts: list[str] = []
    failed = 0
    for i, r in enumerate(results, 1):
        if r.get("ok"):
            parts.append(f"【第 {i} 部分】\n{str(r.get('output', ''))[:3000]}")
        else:
            failed += 1
            parts.append(f"【第 {i} 部分】\n(该部分处理超时/失败: {str(r.get('error', ''))[:100]})")
    head = f"# {title}\n\n" if title else ""
    tail = f"\n\n> 注: {failed} 个分片未完成" if failed else ""
    return head + "\n\n".join(parts) + tail


async def dispatch_parallel(
    prompt: str,
    caller: Callable[[str, int], Awaitable[dict[str, Any]]],
    *,
    max_parallel: int = 4,
    min_chunk_len: int = 200,
    title: str = "",
) -> dict[str, Any]:
    """并行分派: 切分 → gather(caller(chunk, idx)) → 聚合。

    Args:
        caller: 异步调用函数 (chunk, index) -> {"ok", "output"|"error"}
        max_parallel: 并行度 (默认 4)
        min_chunk_len: 低于该长度不切分 (短文直通)
    """
    chunks = split_prompt(prompt, max_chunks=max_parallel, min_len=min_chunk_len)
    if len(chunks) <= 1:
        r = await caller(prompt, 0)
        return {"parallel": False, "chunks": 1, **r,
                "output": r.get("output", ""), "aggregated": r.get("output", "")}

    t0 = asyncio.get_event_loop().time()
    results = await asyncio.gather(
        *[caller(chunk, i) for i, chunk in enumerate(chunks)],
        return_exceptions=True,
    )
    normalized: list[dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            normalized.append({"ok": False, "error": f"{type(r).__name__}: {r}"[:200]})
        elif isinstance(r, dict):
            normalized.append(r)
        else:
            normalized.append({"ok": False, "error": f"caller 返回异常类型: {type(r).__name__}"})

    aggregated = aggregate_results(normalized, title=title)
    return {
        "parallel": True,
        "chunks": len(chunks),
        "elapsed_s": round(asyncio.get_event_loop().time() - t0, 3),
        "ok": any(r.get("ok") for r in normalized),
        "output": aggregated,
        "aggregated": aggregated,
        "partial": [r.get("ok", False) for r in normalized],
    }


__all__ = ["split_prompt", "aggregate_results", "dispatch_parallel"]
