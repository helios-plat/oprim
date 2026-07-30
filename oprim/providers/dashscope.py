"""ProviderRegistry register for qwen3_dashscope (LLM + embedding via DashScope).

B5: the native `dashscope.Generation.call` SDK path returns 400 "Access
denied" under current account billing restrictions; the OpenAI-compatible
endpoint (`/compatible-mode/v1/chat/completions`) is not subject to that
restriction, so the LLM caller goes through it instead. DashScope also
occasionally emits malformed JSON for structured output (numeric ids,
scenes/shots as list[str], bare null) — coerced here once, at the LLM
boundary, so oskill's Pydantic schemas (Storyboard/Script/...) validate.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

_COMPAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

_IMPORTANCE_WORDS = {
    "low": 1,
    "minor": 1,
    "medium": 2,
    "normal": 2,
    "high": 3,
    "major": 3,
    "critical": 4,
    "extreme": 4,
}


def _coerce_llm_json(text: str) -> str:
    """Coerce common DashScope structured-output quirks to schema-compatible JSON."""
    clean_text = text.strip()
    if clean_text.startswith("```"):
        match = re.search(r"```(?:json)?\n?(.*?)\n?```", clean_text, re.DOTALL)
        if match:
            clean_text = match.group(1).strip()

    json_match = re.search(r"(\{.*\}|\[.*\])", clean_text, re.DOTALL)
    if not json_match:
        return text

    try:
        data = json.loads(json_match.group(1))
    except json.JSONDecodeError:
        return text

    def _coerce(obj: Any) -> Any:
        if isinstance(obj, dict):
            res: dict[str, Any] = {}
            for k, v in obj.items():
                if (k.endswith("_id") or k == "id") and isinstance(v, (int, float)):
                    res[k] = str(v)
                elif k in ("importance", "index", "scene_index"):
                    if isinstance(v, (int, float)):
                        res[k] = round(v)
                    elif isinstance(v, str) and v.lower() in _IMPORTANCE_WORDS:
                        res[k] = _IMPORTANCE_WORDS[v.lower()]
                    elif isinstance(v, str):
                        try:
                            res[k] = int(v)
                        except ValueError:
                            res[k] = 0
                    else:
                        res[k] = 0
                elif k in ("scenes", "shots") and isinstance(v, list):
                    fld = "visual_description" if k == "scenes" else "narration"
                    res[k] = [
                        {"id": str(i + 1), fld: item} if isinstance(item, str) else _coerce(item)
                        for i, item in enumerate(v)
                    ]
                elif v is None:
                    res[k] = ""
                else:
                    res[k] = _coerce(v)
            return res
        if isinstance(obj, list):
            return [_coerce(i) for i in obj]
        if obj is None:
            return ""
        return obj

    return json.dumps(_coerce(data), ensure_ascii=False)


class DashScopeLLMResponse:
    """Dict-like + awaitable LLM response.

    oskill callers use both conventions on the same provider:
        result = await llm(messages=...)   # agentic/async pipelines
        content = result.get("content")    # oskill.storyboard_planner (sync)
    The HTTP call happens eagerly in __init__ so both work off one response.
    """

    def __init__(self, resp: dict[str, Any]) -> None:
        self._resp = resp
        choices = resp.get("output", {}).get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            self._resp["content"] = _coerce_llm_json(text)

    def __await__(self) -> Any:
        async def _dummy() -> dict[str, Any]:
            return self._resp

        return _dummy().__await__()

    def get(self, key: str, default: Any = None) -> Any:
        return self._resp.get(key, default)


def _call_compat(
    *,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> dict[str, Any]:
    import httpx

    from oprim.errors import LLMError, LLMRateLimitError

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = httpx.post(
        _COMPAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=120.0,
    )
    data = r.json()
    if r.status_code == 429:
        raise LLMRateLimitError(f"DashScope rate limit: {data.get('error', data)}")
    if r.status_code != 200:
        raise LLMError(f"DashScope error {r.status_code}: {data.get('error', data)}")

    # Adapt OpenAI-compatible response shape -> native {"output": {"choices": [...]}}.
    choices = [
        {"message": c.get("message", {}), "finish_reason": c.get("finish_reason", "")}
        for c in data.get("choices", [])
    ]
    return {"output": {"choices": choices}, "usage": data.get("usage", {})}


def _make_llm_caller(*, default_model: str) -> object:
    from oprim._config import cfg
    from oprim._logging import log as olog
    from oprim.errors import LLMError, LLMRateLimitError

    def caller(
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **_: object,
    ) -> DashScopeLLMResponse:
        api_key = str(cfg.get("DASHSCOPE_API_KEY") or "")
        model_id = model or default_model
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = _call_compat(
                    messages=messages,
                    model=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key,
                )
                return DashScopeLLMResponse(resp)
            except (LLMError, LLMRateLimitError):
                raise
            except Exception as e:
                last_err = e
                if attempt < 2:
                    olog.warning("dashscope llm retry", attempt=attempt, error=str(e))
                    time.sleep(2**attempt)
        raise LLMError(f"qwen3_dashscope failed after 3 retries: {last_err}") from last_err

    return caller


def register(*, replace: bool = False) -> None:
    """Register qwen3_dashscope LLM + embedding into ProviderRegistry.

    Also registers the same OpenAI-compat caller as "llm"/"default" so
    downstream singleton lookups (e.g. omodul's `_default_llm` ->
    `get().generic("llm", "default")`) resolve without hevi needing its
    own inline dashscope monkeypatch.
    """
    from obase import ProviderRegistry

    from oprim.embedding.qwen3_dashscope import Qwen3DashscopeEmbedder

    caller = _make_llm_caller(default_model="qwen-plus")
    ProviderRegistry.register("llm", "qwen3_dashscope", caller, replace=replace)
    ProviderRegistry.register("llm", "default", caller, replace=replace)
    embedder = Qwen3DashscopeEmbedder()
    ProviderRegistry.register("embedding", "qwen3_dashscope", embedder.embed, replace=replace)
