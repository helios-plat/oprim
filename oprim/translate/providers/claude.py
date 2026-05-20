"""Claude translation provider (Anthropic API, async)."""
from __future__ import annotations

import asyncio
import time

from oprim._config import cfg
from oprim._logging import log
from oprim.errors import LLMError
from oprim.translate.protocol import (
    SYSTEM_PROMPT_TRANSLATE,
    TranslationRequest,
    TranslationResult,
)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_INPUT_COST_PER_1K = 0.003   # $3 / 1M tokens
_OUTPUT_COST_PER_1K = 0.015  # $15 / 1M tokens


class ClaudeProvider:
    """TranslationProvider backed by Anthropic API (async AsyncAnthropic)."""

    @property
    def name(self) -> str:
        return "claude"

    async def translate(self, request: TranslationRequest) -> TranslationResult:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("anthropic package not installed") from e

        api_key = cfg.get("ANTHROPIC_API_KEY")
        client = anthropic.AsyncAnthropic(api_key=str(api_key) if api_key else None)
        model_id = request.model or _DEFAULT_MODEL
        user_msg = (
            f"将以下 {request.source_lang} 文本翻译为 {request.target_lang}：\n\n{request.text}"
        )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                t0 = time.monotonic()
                resp = await client.messages.create(
                    model=model_id,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT_TRANSLATE,
                    messages=[{"role": "user", "content": user_msg}],
                )
                elapsed = time.monotonic() - t0
                translated = resp.content[0].text
                in_tok = resp.usage.input_tokens
                out_tok = resp.usage.output_tokens
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
                if attempt < 2:
                    log.warning("claude.translate_retry", attempt=attempt, error=str(e))
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"Claude translate failed after 3 retries: {last_err}") from last_err

    async def health_check(self) -> bool:
        try:
            import anthropic
            api_key = cfg.get("ANTHROPIC_API_KEY")
            client = anthropic.AsyncAnthropic(api_key=str(api_key) if api_key else None)
            resp = await client.messages.create(
                model=_DEFAULT_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return bool(resp.content)
        except Exception as e:
            log.warning("claude.health_check_failed", error=str(e))
            return False

    def estimate_cost(self, char_count: int) -> float:
        approx_tokens = char_count / 3
        return (approx_tokens * _INPUT_COST_PER_1K + approx_tokens * _OUTPUT_COST_PER_1K) / 1000
