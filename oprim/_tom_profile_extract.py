"""oprim.tom_profile_extract — 人类用户心智模型 (ToM) 偏好提取.

单次从历史对话提炼用户的沟通偏好、风险倾向与技术审美（LLM Protocol 注入）。

Example:
    >>> r = tom_profile_extract("用户对话...", llm_caller=llm)
    >>> r["status"]
    'success'
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class TomExtractError(OprimError):
    """ToM 提取失败。"""


@runtime_checkable
class LLMCallerProtocol(Protocol):
    """LLM 调用协议（prompt 风格注入面）。"""

    def __call__(self, prompt: str, *, temperature: float = 0.0) -> str: ...


def tom_profile_extract(
    dialogue_history: str,
    *,
    llm_caller: LLMCallerProtocol,
) -> dict[str, Any]:
    """从对话历史提取 ToM 画像。

    Args:
        dialogue_history: 历史对话文本。
        llm_caller: LLM 调用器（注入）。

    Returns:
        {"status": "success", "tom_profile": dict}
        解析失败时 status="failed" + error（不 raise）。

    Raises:
        OprimValidationError: dialogue_history 为空 / llm_caller 未注入。
    """
    if not dialogue_history or not dialogue_history.strip():
        raise OprimValidationError("tom_profile_extract: dialogue_history must not be empty")
    if llm_caller is None:
        raise OprimValidationError("tom_profile_extract: llm_caller must be injected")

    prompt = (
        "Analyze the following user dialogue history and extract Theory of Mind (ToM) profile:\n"
        f"{dialogue_history[:6000]}\n"
        "Reply in JSON: {\n"
        '  "risk_aversion": "high/medium/low",\n'
        '  "communication_style": "direct/concise/detailed",\n'
        '  "technical_preference": "conservative/cutting-edge"\n'
        "}"
    )
    try:
        raw = llm_caller(prompt, temperature=0.0)
    except Exception as exc:
        raise TomExtractError(f"tom_profile_extract: LLM call failed: {exc}", cause=exc) from exc

    try:
        profile = json.loads(raw if isinstance(raw, str) else str(raw))
    except Exception as exc:
        return {"status": "failed", "tom_profile": {}, "error": f"invalid JSON: {exc}"}

    return {"status": "success", "tom_profile": profile}
