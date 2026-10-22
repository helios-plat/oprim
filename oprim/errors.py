"""Stratum-domain error hierarchy for oprim knowledge-management sub-packages."""

from __future__ import annotations

__oprim_layer__ = "infra"  # shared base module, not a capability
__oprim_exports__ = [
    "ConfigError",
    "DuplicateSubstrateError",
    "EmbeddingError",
    "FulltextError",
    "IngestError",
    "LLMError",
    "LLMRateLimitError",
    "MetaDBError",
    "PDFParseError",
    "QuotaExceededError",
    "StratumError",
    "UnsupportedFileTypeError",
    "UnsupportedImageError",
    "VectorDBError",
    "annotations",
]


class StratumError(Exception):
    """Base class for all Stratum errors."""


class ConfigError(StratumError):
    """Configuration error."""


class PDFParseError(StratumError):
    """PDF parsing failed."""


class UnsupportedFileTypeError(StratumError):
    """File type not supported for this operation."""


class UnsupportedImageError(StratumError):
    """Image type or format not supported."""


class EmbeddingError(StratumError):
    """Embedding generation failed."""


class QuotaExceededError(EmbeddingError):
    """API quota exceeded."""


class VectorDBError(StratumError):
    """Vector database operation failed."""


class FulltextError(StratumError):
    """Fulltext index operation failed."""


class MetaDBError(StratumError):
    """Metadata database operation failed."""


class LLMError(StratumError):
    """LLM call failed."""


class LLMRateLimitError(LLMError):
    """LLM rate limit hit."""


class IngestError(StratumError):
    """Ingest pipeline error."""


class DuplicateSubstrateError(IngestError):
    """Substrate already exists (duplicate detected)."""
