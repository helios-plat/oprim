"""Terminology glossary — preserve domain-specific terms across translation."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class GlossaryEntry:
    source_term: str
    target_term: str
    source_lang: str
    target_lang: str


class TerminologyGlossary:
    """Maps source-language terms to target-language equivalents.

    Usage pattern:
        1. glossary.apply_to_system_prompt(system_prompt) — inject rules into the system
           prompt so the LLM respects terminology.
        2. Alternatively, use protect() / restore() for token-swap approach.

    The system-prompt injection approach is simpler and works for all providers.
    """

    def __init__(self) -> None:
        self._entries: list[GlossaryEntry] = []

    def add(self, source_term: str, target_term: str, source_lang: str, target_lang: str) -> None:
        self._entries.append(
            GlossaryEntry(
                source_term=source_term,
                target_term=target_term,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )

    def entries_for(self, source_lang: str, target_lang: str) -> list[GlossaryEntry]:
        return [
            e for e in self._entries
            if e.source_lang == source_lang and e.target_lang == target_lang
        ]

    def build_system_addendum(self, source_lang: str, target_lang: str) -> str:
        """Return extra system-prompt text enforcing glossary terms.

        Returns empty string if no matching entries.
        """
        entries = self.entries_for(source_lang, target_lang)
        if not entries:
            return ""
        lines = ["术语表（必须按以下对应翻译，不得更改）:"]
        for e in entries:
            lines.append(f"  {e.source_term} → {e.target_term}")
        return "\n".join(lines)

    def protect(self, text: str, source_lang: str, target_lang: str) -> tuple[str, dict[str, str]]:
        """Replace source terms with unique placeholders.

        Returns (protected_text, token_map) where token_map maps placeholder → target_term.
        Call restore(translated_text, token_map) afterward.
        """
        entries = self.entries_for(source_lang, target_lang)
        token_map: dict[str, str] = {}
        protected = text
        for i, entry in enumerate(entries):
            token = f"__TERM_{i:04d}__"
            protected = re.sub(
                re.escape(entry.source_term),
                token,
                protected,
                flags=re.IGNORECASE,
            )
            token_map[token] = entry.target_term
        return protected, token_map

    def restore(self, translated: str, token_map: dict[str, str]) -> str:
        """Replace placeholders with their target terms."""
        result = translated
        for token, target_term in token_map.items():
            result = result.replace(token, target_term)
        return result

    def __len__(self) -> int:
        return len(self._entries)
