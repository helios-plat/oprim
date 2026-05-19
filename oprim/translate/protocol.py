"""TranslationProvider Protocol + shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SYSTEM_PROMPT_TRANSLATE = (
    "你是专业翻译，严格遵守以下规则：\n"
    "1. 只输出翻译内容，不要任何解释、说明或注释。\n"
    "2. 保留代码标识符（变量名、函数名、类名等）不翻译。\n"
    "3. 保留学术引用格式（如 [1]、(Author, Year)、stratum://... 等）。\n"
    "4. 保留 Markdown 格式（标题符号、**加粗**、`代码` 等）。\n"
    "5. 只输出翻译结果，不要重复原文。"
)


@dataclass
class TranslationRequest:
    text: str
    source_lang: str  # e.g. "en", "zh"
    target_lang: str  # e.g. "zh", "en"
    domain: str | None = None  # "academic", "literary", "technical"
    model: str | None = None  # provider-specific model override


@dataclass
class TranslationResult:
    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    source_lang: str
    target_lang: str


@runtime_checkable
class TranslationProvider(Protocol):
    """Protocol all translation providers must implement."""

    @property
    def name(self) -> str: ...

    def translate(self, request: TranslationRequest) -> TranslationResult: ...

    def estimate_cost(self, char_count: int) -> float:
        """Estimate cost in USD for translating char_count characters."""
        ...
