"""TranslationRouter — route requests to providers by quality / preference."""
from __future__ import annotations

from oprim._logging import log
from oprim.translate.protocol import TranslationProvider, TranslationRequest


class TranslationRouter:
    """Routes a TranslationRequest to the most appropriate provider.

    Routing rules (evaluated in order):
    1. quality="premium"            → claude (if registered)
    2. user_preference="domestic"   → qwen3 (if registered)
    3. default                      → deepseek

    DeepSeek is always required as the fallback; a ValueError is raised if it
    is absent.
    """

    def __init__(self, providers: dict[str, TranslationProvider]) -> None:
        if "deepseek" not in providers:
            raise ValueError("TranslationRouter requires 'deepseek' provider as default")
        self.providers = providers

    def route(self, request: TranslationRequest) -> TranslationProvider:
        """Return the provider selected for this request."""
        if request.quality == "premium" and "claude" in self.providers:
            selected = "claude"
        elif request.user_preference == "domestic" and "qwen3" in self.providers:
            selected = "qwen3"
        else:
            selected = "deepseek"

        log.info(
            "translate.router.selected",
            provider=selected,
            quality=request.quality,
            preference=request.user_preference,
        )
        return self.providers[selected]

    def provider_names(self) -> list[str]:
        return list(self.providers.keys())
