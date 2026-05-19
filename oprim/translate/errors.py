"""Translation-specific exception hierarchy."""
from __future__ import annotations

from oprim.errors import StratumError


class TranslationError(StratumError):
    """Base class for all oprim.translate exceptions."""


class ProviderUnavailableError(TranslationError):
    """Provider API unavailable — key missing, network failure, or sustained errors."""


class TokenLimitExceededError(TranslationError):
    """Input text exceeds provider's max_input_tokens."""


class CheckpointCorruptedError(TranslationError):
    """Checkpoint file exists but cannot be parsed or is structurally invalid."""


class FormatPreservationError(TranslationError):
    """Post-translation format validation failed (markdown structure damaged, char count mismatch)."""


class ChunkingError(TranslationError):
    """Chunker produced zero chunks or invalid chunk sequence."""
