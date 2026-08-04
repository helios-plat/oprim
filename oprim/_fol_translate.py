"""oprim.fol_translate — 一阶谓词逻辑 (FOL) 转换原子操作.

单次把自然语言命题经注入的 LLM 转化为 FOL 表达式 + Z3 语法声明。

Example:
    >>> r = await fol_translate("所有人都会死", llm_caller=llm)
    >>> r["status"]
    'success'
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from oprim._exceptions import OprimError, OprimValidationError


class FolTranslateError(OprimError):
    """FOL 转换失败。"""


@runtime_checkable
class LLMCallerProtocol(Protocol):
    """LLM 调用协议（prompt 风格注入面）。"""

    def __call__(self, prompt: str, *, temperature: float = 0.0) -> str: ...


def fol_translate(
    sentence: str,
    *,
    llm_caller: LLMCallerProtocol,
) -> dict[str, Any]:
    """单次将自然语言命题转化为一阶谓词逻辑表达式与 Z3 语法声明。

    Args:
        sentence: 自然语言命题。
        llm_caller: LLM 调用器（注入，prompt 风格协议）。

    Returns:
        {"status": "success", "declarations": [{"name", "type"}],
         "constraints": [str]}
        失败时 status="error" + error 字段（不 raise）。

    Raises:
        OprimValidationError: sentence 为空 / llm_caller 未注入。
    """
    if not sentence or not sentence.strip():
        raise OprimValidationError("fol_translate: sentence must not be empty")
    if llm_caller is None:
        raise OprimValidationError("fol_translate: llm_caller must be injected")

    prompt = (
        "Translate the following statement into First-Order Logic (FOL) and Z3 constraints.\n"
        f"Statement: {sentence}\n"
        "Output JSON format: {\n"
        '  "declarations": [{"name": "p", "type": "Bool"}],\n'
        '  "constraints": ["Implies(p, q)"]\n'
        "}"
    )
    try:
        raw_res = llm_caller(prompt, temperature=0.0)
    except Exception as exc:
        raise FolTranslateError(
            f"fol_translate: LLM call failed: {exc}", cause=exc
        ) from exc

    try:
        parsed = json.loads(raw_res if isinstance(raw_res, str) else str(raw_res))
    except Exception as exc:
        return {
            "status": "error",
            "error": f"invalid JSON from LLM: {exc}",
            "declarations": [],
            "constraints": [],
        }

    return {
        "status": "success",
        "declarations": parsed.get("declarations", []),
        "constraints": parsed.get("constraints", []),
    }
