"""Errors for oprim.external sub-package."""
from __future__ import annotations

from oprim.errors import StratumError


class ExternalToolError(StratumError):
    """Generic external tool failure."""


class ExternalToolTimeout(ExternalToolError):
    """External tool call timed out."""


class ExternalToolUnavailable(ExternalToolError):
    """External tool is unreachable or not configured."""


class CircuitBreakerOpen(ExternalToolError):
    """Circuit breaker is open — call rejected."""
