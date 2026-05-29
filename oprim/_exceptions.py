"""oprim exception hierarchy — all oprim errors inherit from OprimError."""

from __future__ import annotations


class OprimError(Exception):
    """Base class for all oprim exceptions."""


class OprimConnectionError(OprimError):
    """External connection failure (docker daemon / pg / rabbitmq / http / etc.)."""


class OprimTimeoutError(OprimError):
    """Operation timed out."""


class OprimNotFoundError(OprimError):
    """Target object not found (container_id missing, queue absent, etc.)."""


class OprimAuthError(OprimError):
    """Authentication failure (S3 / DB / Caddy admin / etc.)."""


class OprimValidationError(OprimError):
    """Input parameter validation failed."""
