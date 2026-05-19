"""Translation provider registry."""
from __future__ import annotations

from oprim.translate.providers.claude import ClaudeProvider
from oprim.translate.providers.deepseek import DeepSeekProvider
from oprim.translate.providers.gemini import GeminiProvider
from oprim.translate.providers.qwen3 import Qwen3Provider
from oprim.translate.protocol import TranslationProvider

_REGISTRY: dict[str, type[TranslationProvider]] = {
    "deepseek": DeepSeekProvider,
    "claude": ClaudeProvider,
    "qwen3": Qwen3Provider,
    "gemini": GeminiProvider,
}


def get_provider(name: str) -> TranslationProvider:
    """Instantiate a provider by name.

    Raises:
        ValueError: Unknown provider name.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown translation provider: {name!r}. Choose from {list(_REGISTRY)}")
    return cls()


__all__ = [
    "DeepSeekProvider",
    "ClaudeProvider",
    "Qwen3Provider",
    "GeminiProvider",
    "get_provider",
]
