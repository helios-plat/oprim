"""oprim.embedding_gen — 单次文本/代码向量生成.

薄组合 oprim.embed_text（单次嵌入调用），输出统一 dict 结构。

Example:
    >>> r = await embedding_gen("hello world", caller=embed_caller)
    >>> len(r["vector"])
    1536
"""

from __future__ import annotations

from typing import Any

from oprim.embed_text import embed_text


async def embedding_gen(
    text: str,
    *,
    caller: Any,
    model: str = "text-embedding-3-small",
) -> dict[str, Any]:
    """单次文本/代码向量生成。

    Args:
        text: 待嵌入文本（代码片段亦可）。
        caller: EmbedCaller Protocol 实例（见 oprim._protocols.EmbedCaller）。
        model: 嵌入模型名。

    Returns:
        {"vector": list[float], "model": str, "token_count": int, "dim": int}

    Raises:
        LLMOprimError: 文本为空或嵌入调用失败。
    """
    result = await embed_text(text, caller=caller, model=model)
    return {
        "vector": list(result.vector),
        "model": result.model,
        "token_count": result.token_count,
        "dim": len(result.vector),
    }
