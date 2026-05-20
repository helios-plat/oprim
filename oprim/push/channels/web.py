"""Web Push channel using VAPID (RFC 8030 / RFC 8292).

Dependencies: pywebpush (optional — push gracefully disabled if not installed).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from oprim._logging import log
from oprim.push.errors import PushConfigError, PushDeliveryError
from oprim.push.protocol import PushResult

_PYWEBPUSH_AVAILABLE = False
try:
    from pywebpush import webpush, WebPushException  # type: ignore[import]
    _PYWEBPUSH_AVAILABLE = True
except ImportError:
    def webpush(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("pywebpush is not installed; run: pip install pywebpush")

    class WebPushException(Exception):  # type: ignore[no-redef]
        pass


class WebPushChannel:
    """Sends push notifications to browser endpoints via the Web Push protocol."""

    name = "web"

    def __init__(
        self,
        vapid_private_key: str,
        vapid_claims: dict,
    ) -> None:
        if not vapid_private_key:
            raise PushConfigError("vapid_private_key is required for WebPushChannel")
        if not vapid_claims.get("sub"):
            raise PushConfigError("vapid_claims must contain 'sub' (mailto: or https:)")
        self._vapid_private_key = vapid_private_key
        self._vapid_claims = vapid_claims

    async def send(
        self,
        recipient: str,
        title: str,
        body: str,
        deep_link: str | None = None,
        metadata: dict | None = None,
    ) -> PushResult:
        """Send a Web Push notification.

        recipient: JSON string of the browser push subscription object
        (keys: endpoint, keys.p256dh, keys.auth)
        """
        if not _PYWEBPUSH_AVAILABLE:
            log.warning("web_push_disabled", reason="pywebpush not installed")
            return PushResult(
                channel=self.name,
                success=False,
                recipient=recipient[:50],
                error_message="pywebpush not installed",
            )

        try:
            subscription = json.loads(recipient) if isinstance(recipient, str) else recipient
        except json.JSONDecodeError as exc:
            return PushResult(
                channel=self.name,
                success=False,
                recipient=recipient[:50],
                error_message=f"Invalid subscription JSON: {exc}",
            )

        payload = json.dumps({
            "title": title,
            "body": body,
            "deep_link": deep_link,
            **(metadata or {}),
        })

        def _send():
            return webpush(
                subscription_info=subscription,
                data=payload,
                vapid_private_key=self._vapid_private_key,
                vapid_claims=self._vapid_claims,
            )

        try:
            await asyncio.to_thread(_send)
            log.info("web_push_sent", recipient=recipient[:30])
            return PushResult(
                channel=self.name,
                success=True,
                recipient=recipient[:50],
                sent_at=datetime.now(tz=timezone.utc),
            )
        except Exception as exc:
            log.error("web_push_failed", error=str(exc), recipient=recipient[:30])
            return PushResult(
                channel=self.name,
                success=False,
                recipient=recipient[:50],
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        return _PYWEBPUSH_AVAILABLE and bool(self._vapid_private_key)
