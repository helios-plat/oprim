"""Translation provider registry (Phase 10: DeepSeek + Claude + Qwen3)."""
from __future__ import annotations

from oprim.translate.providers.claude import ClaudeProvider
from oprim.translate.providers.deepseek import DeepSeekProvider
from oprim.translate.providers.qwen3 import Qwen3Provider
from oprim.translate.protocol import TranslationProvider

_REGISTRY: dict[str, type[TranslationProvider]] = {
    "deepseek": DeepSeekProvider,
    "claude": ClaudeProvider,
    "qwen3": Qwen3Provider,
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


def build_all_providers() -> dict[str, TranslationProvider]:
    """Build all registered providers (for use with TranslationRouter)."""
    return {name: cls() for name, cls in _REGISTRY.items()}


__all__ = [
    "DeepSeekProvider",
    "ClaudeProvider",
    "Qwen3Provider",
    "get_provider",
    "build_all_providers",
]
