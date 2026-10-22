"""oprim.llm_chat_call — 单次 LLM/VLM API 基础调用（返回标准化 dict，含 usage）.

薄组合 oprim.llm_complete（消息校验 + token 预算 + 错误标准化 + usage 提取），
输出统一 dict 结构，供 oskill/omodul 直接消费。

Example:
    >>> resp = await llm_chat_call(
    ...     [{"role": "user", "content": "hi"}],
    ...     caller=my_caller,
    ...     model="claude-sonnet-4-6",
    ... )
    >>> resp["usage"]["input_tokens"] >= 0
    True
"""

from __future__ import annotations

from typing import Any

from oprim._exceptions import LLMOprimError
from oprim.llm._llm_complete import llm_complete


async def llm_chat_call(
    messages: list[dict[str, Any]],
    *,
    caller: Any,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    tools: list[dict] | None = None,
    system: str | None = None,
    budget_tokens: int | None = None,
) -> dict[str, Any]:
    """单次 LLM/VLM 聊天补全，返回标准化 dict。

    Args:
        messages: OpenAI 风格消息列表 [{"role", "content"}, ...]。
        caller: LLMCaller Protocol 实例（由调用方注入，见 oprim._protocols / obase）。
        model: 模型名（用于成本估算与 token 计数）。
        max_tokens: 最大输出 token 数。
        tools: 工具 schema 列表（可选）。
        system: system prompt（可选）。
        budget_tokens: 输入 token 预算；超出抛 BudgetExceededError。

    Returns:
        {
            "text": str,
            "tool_calls": list[dict],
            "stop_reason": str,
            "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
            "cost_usd": float,
            "model": str,
        }

    Raises:
        LLMOprimError: 消息格式错误 / provider 调用失败。
    """
    if not messages:
        raise LLMOprimError("llm_chat_call: messages must not be empty")

    response = await llm_complete(
        messages,
        caller=caller,
        tools=tools,
        max_tokens=max_tokens,
        system=system,
        budget_tokens=budget_tokens,
        model=model,
    )

    return {
        "text": response.text,
        "tool_calls": list(response.tool_calls),
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        },
        "cost_usd": round(response.cost_usd, 6),
        "model": model,
    }
