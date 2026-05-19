"""DeepSeek translation provider (OpenAI-compatible API)."""
from __future__ import annotations

import time

from oprim._config import cfg
from oprim._logging import log
from oprim.errors import LLMError, LLMRateLimitError
from oprim.translate.protocol import (
    SYSTEM_PROMPT_TRANSLATE,
    TranslationRequest,
    TranslationResult,
)

_DEFAULT_MODEL = "deepseek-chat"
_INPUT_COST_PER_1K = 0.00027   # $0.27 / 1M tokens
_OUTPUT_COST_PER_1K = 0.00110  # $1.10 / 1M tokens


class DeepSeekProvider:
    """TranslationProvider backed by api.deepseek.com (OpenAI-compatible)."""

    @property
    def name(self) -> str:
        return "deepseek"

    def translate(self, request: TranslationRequest) -> TranslationResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMError("openai package not installed") from e

        api_key = cfg.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMError("DEEPSEEK_API_KEY not configured")

        client = OpenAI(api_key=str(api_key), base_url="https://api.deepseek.com")
        model_id = request.model or _DEFAULT_MODEL
        user_msg = (
            f"将以下 {request.source_lang} 文本翻译为 {request.target_lang}：\n\n{request.text}"
        )

        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_TRANSLATE},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=4096,
                )
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
                )
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str:
                    raise LLMRateLimitError(f"DeepSeek rate limit: {e}") from e
                if attempt < 2:
                    log.warning("deepseek.translate_retry", attempt=attempt, error=str(e))
                    time.sleep(2**attempt)
        raise LLMError(f"DeepSeek translate failed after 3 retries: {last_err}") from last_err

    def estimate_cost(self, char_count: int) -> float:
        approx_tokens = char_count / 3  # ~3 chars per token for mixed CJK/Latin
        return (approx_tokens * _INPUT_COST_PER_1K + approx_tokens * _OUTPUT_COST_PER_1K) / 1000
