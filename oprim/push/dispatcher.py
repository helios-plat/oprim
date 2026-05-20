"""Push dispatcher — routes notifications across multiple channels with priority fallback."""
from __future__ import annotations

from oprim._logging import log
from oprim.meta_db.duckdb import MetaDB
from oprim.push.errors import PushError
from oprim.push.protocol import PushChannel, PushResult


class PushDispatcher:
    """Routes push notifications to registered channels with priority ordering.

    Attempts each channel in preference order; stops after first success.
    Falls through all channels on failure, returning all results.
    """

    def __init__(
        self,
        channels: dict[str, PushChannel],
        db: MetaDB | None = None,
    ) -> None:
        self._channels = channels
        self._db = db

    async def push(
        self,
        user_id: str,
        title: str,
        body: str,
        channels_preference: list[str] | None = None,
        deep_link: str | None = None,
        metadata: dict | None = None,
    ) -> list[PushResult]:
        """Send notification to user, trying channels in preference order.

        Returns all PushResult objects (including failures). Stops after first success.
        """
        if channels_preference is None:
            channels_preference = ["web", "email"]

        results: list[PushResult] = []

        for ch_name in channels_preference:
            if ch_name not in self._channels:
                log.warning("push_channel_not_registered", channel=ch_name)
                continue

            channel = self._channels[ch_name]
            recipient = await self._get_recipient(user_id, ch_name)
            if not recipient:
                log.warning("push_no_recipient", user_id=user_id, channel=ch_name)
                continue

            try:
                result = await channel.send(
                    recipient=recipient,
                    title=title,
                    body=body,
                    deep_link=deep_link,
                    metadata=metadata,
                )
            except PushError as exc:
                log.error("push_channel_error", channel=ch_name, error=str(exc))
                result = PushResult(
                    channel=ch_name, success=False, recipient=recipient, error_message=str(exc)
                )

            results.append(result)
            if result.success:
                log.info("push_sent", user_id=user_id, channel=ch_name)
                break

        if not results:
            log.warning("push_no_channel_available", user_id=user_id)

        return results

    async def _get_recipient(self, user_id: str, channel: str) -> str | None:
        """Look up the user's recipient address for *channel* from push_subscriptions."""
        if self._db is None:
            return None
        rows = self._db.fetchall(
            "SELECT recipient FROM push_subscriptions "
            "WHERE user_id = ? AND channel = ? AND enabled = TRUE LIMIT 1",
            [user_id, channel],
        )
        return rows[0][0] if rows else None

    def register_channel(self, channel: PushChannel) -> None:
        self._channels[channel.name] = channel

    def unregister_channel(self, name: str) -> None:
        self._channels.pop(name, None)
