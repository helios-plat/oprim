"""Changefeed-specific errors."""
from __future__ import annotations

from oprim.errors import StratumError


class ChangefeedError(StratumError):
    """Base class for changefeed errors."""


class ChangefeedConflictError(ChangefeedError):
    """seq conflict — (user_id, seq) already exists."""


class ChangefeedCompactorError(ChangefeedError):
    """Compaction failed; events may not have been modified."""
