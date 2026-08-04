"""Provider-dispatched LLM call with retry and cost tracking."""

from __future__ import annotations

from ._types import LLMResponse
from oprim._config import cfg
from oprim._logging import log as olog
from oprim.errors import LLMError, LLMRateLimitError


def llm_call(
    prompt: str,
    provider: str = "qwen3_dashscope",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system: str | None = None,
) -> LLMResponse:
    if provider == "qwen3_dashscope":
        return _call_dashscope(prompt, model, temperature, max_tokens, system)
    elif provider == "claude":
        return _call_claude(prompt, model, temperature, max_tokens, system)
    elif provider == "nvidia_nim":
        return _call_nvidia_nim(prompt, model, temperature, max_tokens, system)
    elif provider in ("ollama", "ollama_local"):
        return _call_ollama(prompt, model, temperature, max_tokens, system)
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")


def _call_dashscope(prompt, model, temperature, max_tokens, system):
    try:
        import httpx
        import os
    except ImportError as e:
        raise LLMError(f"httpx not installed: {e}")

    api_key = cfg.get("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise LLMError("DASHSCOPE_API_KEY not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    _model = model or "qwen-plus"
    try:
        resp = httpx.post(
            "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _model,
                "input": {"messages": messages},
                "parameters": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["output"]["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=_model)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise LLMRateLimitError(f"DashScope rate limit: {e}")
        raise LLMError(f"DashScope HTTP error {e.response.status_code}: {e}")
    except Exception as e:
        raise LLMError(f"DashScope call failed: {e}")


def _call_claude(prompt, model, temperature, max_tokens, system):
    # Claude 走 obase.ProviderRegistry，此处保留 stub 待接入
    raise LLMError(
        "Claude provider not yet implemented via llm_call; use oprim.llm_complete instead"
    )


def _call_nvidia_nim(prompt, model, temperature, max_tokens, system):
    """NVIDIA NIM — 云端 OpenAI 兼容端点。之前 provider="nvidia_nim" 调这个函数时
    压根没实现(直接落进 llm_call() 的 else 分支报 "Unknown LLM provider"), 不是
    配错了名字——调用方(aii_feedback_service.py 等)一直以为这条路能通。这里补上
    真正的实现, 照抄 aii/aii/aii/api/_provider.py 里已经在用、已验证工作的
    NIM caller 的请求/响应形状(标准 OpenAI chat/completions 格式, 和 DashScope
    自家那套 input/output 包法不是一回事)。
    """
    try:
        import os

        import httpx
    except ImportError as e:
        raise LLMError(f"httpx not installed: {e}") from e

    api_key = cfg.get("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_NIM_API_KEY", "")
    if not api_key:
        raise LLMError("NVIDIA_NIM_API_KEY not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    _model = model or os.getenv("NIM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")
    try:
        resp = httpx.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=_model,
            stop_reason=choice.get("finish_reason", ""),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise LLMRateLimitError(f"NVIDIA NIM rate limit: {e}") from e
        raise LLMError(f"NVIDIA NIM HTTP error {e.response.status_code}: {e}") from e
    except Exception as e:
        raise LLMError(f"NVIDIA NIM call failed: {e}") from e


def _call_ollama(prompt, model, temperature, max_tokens, system):
    """Local Ollama — OpenAI 兼容 /api/chat, 无外部凭证依赖 (本地兜底)."""
    try:
        import os

        import httpx
    except ImportError as e:
        raise LLMError(f"httpx not installed: {e}") from e

    base = (cfg.get("OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL")
            or "http://127.0.0.1:11434").rstrip("/")
    _model = model or os.getenv("OLLAMA_MODEL", "qwen3-8b")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    try:
        # think=False: qwen3 思考 token 会吃光 num_predict 预算 → 200 但空 content。
        # 空 content 再重试兜底。
        for attempt in range(3):
            resp = httpx.post(
                f"{base}/api/chat",
                json={
                    "model": _model,
                    "messages": messages,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
                timeout=180.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("message") or {}).get("content", "") or ""
            if text.strip() or attempt == 2:
                break
        return LLMResponse(
            text=text,
            model=_model,
            stop_reason="stop",
            input_tokens=(data.get("prompt_eval_count") or 0),
            output_tokens=(data.get("eval_count") or 0),
            raw=data,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise LLMRateLimitError(f"Ollama rate limit: {e}") from e
        raise LLMError(f"Ollama HTTP error {e.response.status_code}: {e}") from e
    except Exception as e:
        raise LLMError(f"Ollama call failed: {e}") from e
