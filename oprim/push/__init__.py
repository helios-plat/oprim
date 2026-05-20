"""oprim.push — multi-channel push notification dispatcher (Phase 2)."""
from oprim.push.channels.email import EmailPushChannel
from oprim.push.channels.web import WebPushChannel
from oprim.push.dispatcher import PushDispatcher
from oprim.push.errors import PushConfigError, PushDeliveryError, PushError, PushRateLimitError
from oprim.push.protocol import PushChannel, PushResult

__all__ = [
    "PushDispatcher",
    "PushChannel",
    "PushResult",
    "WebPushChannel",
    "EmailPushChannel",
    "PushError",
    "PushConfigError",
    "PushDeliveryError",
    "PushRateLimitError",
]
