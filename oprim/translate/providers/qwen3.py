"""Qwen3 translation provider (DashScope OpenAI-compatible API, async)."""
from __future__ import annotations

import asyncio
import time

from oprim._config import cfg
from oprim._logging import log
from oprim.errors import LLMError, LLMRateLimitError
from oprim.translate.protocol import (
    SYSTEM_PROMPT_TRANSLATE,
    TranslationRequest,
    TranslationResult,
)

_DEFAULT_MODEL = "qwen-turbo"
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_INPUT_COST_PER_1K = 0.00005   # $0.05 / 1M tokens (qwen-turbo)
_OUTPUT_COST_PER_1K = 0.00010  # $0.10 / 1M tokens


class Qwen3Provider:
    """TranslationProvider backed by DashScope OpenAI-compatible endpoint (async AsyncOpenAI)."""

    @property
    def name(self) -> str:
        return "qwen3"

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise LLMError("openai package not installed") from e

        api_key = cfg.get("DASHSCOPE_API_KEY") or cfg.get("QWEN_API_KEY")
        if not api_key:
            raise LLMError("DASHSCOPE_API_KEY (or QWEN_API_KEY) not configured")

        client = AsyncOpenAI(api_key=str(api_key), base_url=_BASE_URL)
        model_id = request.model or _DEFAULT_MODEL
        user_msg = (
            f"将以下 {request.source_lang} 文本翻译为 {request.target_lang}：\n\n{request.text}"
        )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                t0 = time.monotonic()
                resp = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TRANSLATE},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
                elapsed = time.monotonic() - t0
                translated = resp.choices[0].message.content or ""
                usage = resp.usage
                in_tok = usage.prompt_tokens if usage else 0
                out_tok = usage.completion_tokens if usage else 0
                cost = (in_tok * _INPUT_COST_PER_1K + out_tok * _OUTPUT_COST_PER_1K) / 1000
                return TranslationResult(
                    text=translated,
                    provider=self.name,
                    model=model_id,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost_usd=cost,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    elapsed_seconds=elapsed,
                )
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    raise LLMRateLimitError(f"Qwen3 rate limit: {e}") from e
                if attempt < 2:
                    log.warning("qwen3.translate_retry", attempt=attempt, error=str(e))
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"Qwen3 translate failed after 3 retries: {last_err}") from last_err

    async def health_check(self) -> bool:
        try:
            from openai import AsyncOpenAI
            api_key = cfg.get("DASHSCOPE_API_KEY") or cfg.get("QWEN_API_KEY")
            if not api_key:
                return False
            client = AsyncOpenAI(api_key=str(api_key), base_url=_BASE_URL)
            resp = await client.chat.completions.create(
                model=_DEFAULT_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(resp.choices)
        except Exception as e:
            log.warning("qwen3.health_check_failed", error=str(e))
            return False

    def estimate_cost(self, char_count: int) -> float:
        approx_tokens = char_count / 3
        return (approx_tokens * _INPUT_COST_PER_1K + approx_tokens * _OUTPUT_COST_PER_1K) / 1000
