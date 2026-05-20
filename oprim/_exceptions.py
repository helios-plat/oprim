"""Shared oprim layer exceptions."""
from __future__ import annotations

from obase.exceptions import OBaseError


class OprimError(OBaseError):  # type: ignore[misc]
    """Base class for oprim layer errors."""

    retryable = True
