"""TranslationProvider Protocol + shared dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

SYSTEM_PROMPT_TRANSLATE = (
    "你是专业翻译，严格遵守以下规则：\n"
    "1. 只输出翻译内容，不要任何解释、说明或注释。\n"
    "2. 保留代码标识符（变量名、函数名、类名等）不翻译。\n"
    "3. 保留学术引用格式（如 [1]、(Author, Year)、stratum://... 等）。\n"
    "4. 保留 Markdown 格式（标题符号、**加粗**、`代码` 等）。\n"
    "5. 只输出翻译结果，不要重复原文。"
)


@dataclass
class TranslationContext:
    """Cross-chunk context passed to each translation call for coherence."""
    previous_chunks_summary: str | None = None
    next_chunk_preview: str | None = None      # first ~100 chars of next chunk
    document_metadata: dict = field(default_factory=dict)
    established_proper_nouns: dict[str, str] = field(default_factory=dict)
    chunk_index: int = 0
    total_chunks: int = 1


@dataclass
class TranslationRequest:
    text: str
    source_lang: str  # e.g. "en", "zh"
    target_lang: str  # e.g. "zh", "en"
    domain: str | None = None  # "academic", "literary", "technical"
    model: str | None = None   # provider-specific model override
    quality: Literal["fast", "balanced", "premium"] = "balanced"
    user_preference: Literal["default", "domestic", "international"] = "default"
    context: TranslationContext | None = None


@dataclass
class TranslationResult:
    text: str           # translated text (cleaned of terminology sections)
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    source_lang: str
    target_lang: str
    elapsed_seconds: float = 0.0
    extracted_terminology: dict[str, str] = field(default_factory=dict)
    detected_proper_nouns: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class TranslationProvider(Protocol):
    """Protocol all translation providers must implement (async interface)."""

    @property
    def name(self) -> str: ...

    async def translate(self, request: TranslationRequest) -> TranslationResult: ...

    async def health_check(self) -> bool: ...

    def estimate_cost(self, char_count: int) -> float:
        """Estimate cost in USD for translating char_count characters."""
        ...
