"""oprim.boss_llm_callers — planning / verification LLM calls. Caller injected."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

BossCaller = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


async def call_llm_for_planning(
    messages: list[dict[str, Any]],
    *,
    caller: BossCaller,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """One planning call. Returns a dict that should contain tasks[]. No vendor."""
    raw = await _invoke(caller, messages, max_tokens)
    parsed = _extract_json(raw)
    out = dict(raw) if isinstance(raw, dict) else {"ok": True}
    if parsed:
        out.update(parsed)
    if "tasks" not in out and isinstance(out.get("graph"), dict):
        tasks = out["graph"].get("tasks")
        if isinstance(tasks, list):
            out["tasks"] = tasks
    out.setdefault("ok", True)
    return out


async def call_llm_for_intent(
    messages: list[dict[str, Any]],
    *,
    caller: BossCaller,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """One triage call. Returns action + interpretation. No vendor."""
    raw = await _invoke(caller, messages, max_tokens)
    parsed = _extract_json(raw) if "action" not in raw else {}
    out = dict(raw) if isinstance(raw, dict) else {"ok": True}
    if parsed:
        out.update(parsed)
    action = str(out.get("action") or "ask").strip().lower()
    if action not in {"plan", "ask", "refuse"}:
        action = "ask"
    out["action"] = action
    out.setdefault("interpretation", "")
    out.setdefault("ok", True)
    return out


async def call_llm_for_verification(
    messages: list[dict[str, Any]],
    *,
    caller: BossCaller,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """One QA call. Returns passed + reasoning. No vendor."""
    raw = await _invoke(caller, messages, max_tokens)
    parsed = _extract_json(raw)
    out = dict(raw) if isinstance(raw, dict) else {"ok": True}
    if parsed:
        out.update(parsed)
    out["passed"] = _as_bool(out.get("passed"), default=False)
    if "reasoning" not in out:
        out["reasoning"] = str(out.get("summary") or out.get("content") or "")
    out.setdefault("ok", True)
    return out


async def _invoke(
    caller: BossCaller,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    result = caller(messages=messages, max_tokens=max_tokens)
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[misc]
    if not isinstance(result, dict):
        return {"ok": False, "error": "caller returned non-dict", "content": result}
    out = dict(result)
    out.setdefault("ok", True)
    return out


def _extract_json(raw: dict[str, Any]) -> dict[str, Any]:
    if any(key in raw for key in ("tasks", "passed", "graph")):
        return {}
    text = raw.get("content") or raw.get("text") or raw.get("output") or ""
    if not isinstance(text, str) or not text.strip():
        return {}
    blob = _json_blob(text)
    if blob is None:
        return {}
    return blob


def _json_blob(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = "\n".join(lines[1:])
        if inner.rstrip().endswith("```"):
            inner = inner.rstrip()[:-3]
        stripped = inner.strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].lstrip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "pass", "passed"}:
            return True
        if lowered in {"false", "0", "no", "fail", "failed"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default
