"""Provider-dispatched LLM call with retry and cost tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass

from oprim._config import cfg
from oprim._logging import log as olog
from oprim.errors import LLMError, LLMRateLimitError


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


def llm_call(
    prompt: str,
    provider: str = "qwen3_dashscope",
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    system: str | None = None,
) -> LLMResponse:
    """Make an LLM completion call.

    Args:
        prompt: User message.
        provider: "qwen3_dashscope" or "claude".
        model: Override the default model for the provider.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        system: Optional system prompt.

    Raises:
        LLMError: Call failed after retries.
        LLMRateLimitError: Rate limit hit (no retry).
    """
    if provider == "qwen3_dashscope":
        return _call_dashscope(prompt, model, temperature, max_tokens, system)
    elif provider == "claude":
        return _call_claude(prompt, model, temperature, max_tokens, system)
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")


def _call_dashscope(
    prompt: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    system: str | None,
) -> LLMResponse:
    import dashscope
    from dashscope import Generation

    api_key = cfg.get("DASHSCOPE_API_KEY")
    if api_key:
        dashscope.api_key = str(api_key)

    model_id = model or "qwen-plus"
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = Generation.call(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                result_format="message",
            )
            if resp.status_code == 200:
                text = resp.output.choices[0].message.content
                usage = resp.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                # qwen-plus approximate pricing
                cost = (input_tokens * 0.0004 + output_tokens * 0.0012) / 1000
                return LLMResponse(
                    text=text,
                    model=model_id,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                )
            elif resp.status_code == 429:
                raise LLMRateLimitError(f"DashScope rate limit: {resp.message}")
            else:
                raise LLMError(f"DashScope error {resp.status_code}: {resp.message}")
        except (LLMError, LLMRateLimitError):
            raise
        except Exception as e:
            last_err = e
            if attempt < 2:
                wait = 2**attempt
                olog.warning("dashscope llm retry", attempt=attempt, error=str(e))
                time.sleep(wait)
    raise LLMError(f"LLM call failed after 3 retries: {last_err}") from last_err


def _call_claude(
    prompt: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    system: str | None,
) -> LLMResponse:
    try:
        import anthropic
    except ImportError as e:
        raise LLMError("anthropic package not installed") from e

    api_key = cfg.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=str(api_key) if api_key else None)
    model_id = model or "claude-sonnet-4-6"

    kwargs: dict = {
        "model": model_id,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.messages.create(**kwargs)
            text = resp.content[0].text
            input_tokens = resp.usage.input_tokens
            output_tokens = resp.usage.output_tokens
            # Sonnet 4.6 approximate pricing
            cost = (input_tokens * 0.003 + output_tokens * 0.015) / 1000
            return LLMResponse(
                text=text,
                model=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        except Exception as e:
            last_err = e
            if attempt < 2:
                wait = 2**attempt
                olog.warning("claude llm retry", attempt=attempt, error=str(e))
                time.sleep(wait)
    raise LLMError(
        f"Claude LLM call failed after 3 retries: {last_err}"
    ) from last_err
