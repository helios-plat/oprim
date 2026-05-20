"""Push-specific error hierarchy."""
from __future__ import annotations

from oprim.errors import StratumError


class PushError(StratumError):
    """Base class for push notification errors."""


class PushConfigError(PushError):
    """Push channel misconfigured (missing VAPID keys, SMTP config, etc.)."""


class PushDeliveryError(PushError):
    """Message could not be delivered to the recipient endpoint."""


class PushRateLimitError(PushError):
    """Push provider rate limit hit."""
