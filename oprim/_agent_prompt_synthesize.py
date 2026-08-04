"""oprim.agent_prompt_synthesize — Agent 提示词合成.

基于目标 + 上下文（+ 人设）经注入的 LLM 合成 agent system prompt，
返回标准化 dict（含 usage）。

Example:
    >>> r = await agent_prompt_synthesize("做一个数据分析 agent", caller=llm)
    >>> r["prompt"].strip() != ""
    True
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class PromptSynthError(OprimError):
    """提示词合成失败。"""


@runtime_checkable
class PromptSynthCaller(Protocol):
    """LLM 调用协议（对齐 llm_chat_call 的 caller 面）。"""

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        system: str | None = None,
    ) -> dict[str, Any]: ...


async def agent_prompt_synthesize(
    goal: str,
    *,
    caller: PromptSynthCaller,
    context: str = "",
    persona: str = "",
    capabilities: list[str] | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """合成 agent 提示词。

    Args:
        goal: Agent 目标描述。
        caller: LLM 调用器（注入）。
        context: 附加上下文（代码库/任务背景），可选。
        persona: 人设描述，可选。
        capabilities: 能力清单（工具/技能名），可选。
        max_tokens: 输出上限。

    Returns:
        {"status": "ok", "prompt": str, "goal": str, "usage": dict}

    Raises:
        PromptSynthError: LLM 调用失败。
        OprimValidationError: goal / caller 缺失。
    """
    if not goal or not goal.strip():
        raise OprimValidationError("agent_prompt_synthesize: goal must not be empty")
    if caller is None:
        raise OprimValidationError("agent_prompt_synthesize: caller must be injected")

    sections = [f"## 目标\n{goal}"]
    if persona:
        sections.append(f"## 人设\n{persona}")
    if capabilities:
        sections.append("## 能力\n- " + "\n- ".join(capabilities))
    if context:
        sections.append(f"## 上下文\n{context[:4000]}")

    prompt = (
        "根据以下规格合成一份可直接使用的 agent system prompt"
        "（中文，包含角色、职责边界、工作流、输出规范）：\n\n"
        + "\n\n".join(sections)
    )
    try:
        response = await caller(
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        raise PromptSynthError(
            f"agent_prompt_synthesize: LLM call failed: {exc}", cause=exc
        ) from exc

    content = response.get("content", "")
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return {
        "status": "ok",
        "prompt": str(content).strip(),
        "goal": goal,
        "usage": response.get("usage", {}),
    }
