"""Cursor-based changefeed reader."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from oprim._logging import log
from oprim.meta_db.duckdb import MetaDB
from oprim.changefeed.schema import ChangefeedEvent, EventType


class ChangefeedReader:
    """Reads events from the changefeed_events table in seq order."""

    def __init__(self, db: MetaDB, user_id: str) -> None:
        self.db = db
        self.user_id = user_id

    def read_since(
        self,
        since_seq: int,
        batch_size: int = 100,
        event_types: list[EventType] | None = None,
    ) -> list[ChangefeedEvent]:
        """Return events with seq > since_seq, ascending order.

        This is a synchronous call — DuckDB is in-process and fast.
        """
        if event_types:
            placeholders = ", ".join("?" * len(event_types))
            sql = f"""
                SELECT id, device_id, user_id, event_type, aggregate_id, payload, created_at, seq
                FROM changefeed_events
                WHERE user_id = ? AND seq > ? AND event_type IN ({placeholders})
                ORDER BY seq ASC
                LIMIT ?
            """
            params = [self.user_id, since_seq] + [et.value for et in event_types] + [batch_size]
        else:
            sql = """
                SELECT id, device_id, user_id, event_type, aggregate_id, payload, created_at, seq
                FROM changefeed_events
                WHERE user_id = ? AND seq > ?
                ORDER BY seq ASC
                LIMIT ?
            """
            params = [self.user_id, since_seq, batch_size]

        rows = self.db.fetchall(sql, params)
        events = [_row_to_event(r) for r in rows]
        log.info("changefeed_read_since", since_seq=since_seq, count=len(events), user_id=self.user_id)
        return events

    def get_latest_seq(self) -> int:
        """Return the maximum seq for this user, or 0 if no events."""
        rows = self.db.fetchall(
            "SELECT COALESCE(MAX(seq), 0) FROM changefeed_events WHERE user_id = ?",
            [self.user_id],
        )
        return int(rows[0][0])

    def count(self) -> int:
        """Return total event count for this user."""
        rows = self.db.fetchall(
            "SELECT COUNT(*) FROM changefeed_events WHERE user_id = ?",
            [self.user_id],
        )
        return int(rows[0][0])


def _row_to_event(row: tuple) -> ChangefeedEvent:
    row_id, device_id, user_id, event_type, aggregate_id, payload_str, created_at_raw, seq = row
    payload = json.loads(payload_str) if isinstance(payload_str, str) else payload_str
    if isinstance(created_at_raw, str):
        created_at = datetime.fromisoformat(created_at_raw)
    elif isinstance(created_at_raw, datetime):
        created_at = created_at_raw
    else:
        created_at = datetime.now(tz=timezone.utc)
    return ChangefeedEvent(
        id=int(row_id),
        device_id=device_id,
        user_id=user_id,
        event_type=EventType(event_type),
        aggregate_id=aggregate_id,
        payload=payload,
        created_at=created_at,
        seq=int(seq),
    )
