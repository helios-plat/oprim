"""Changefeed compactor — removes superseded events while preserving history integrity.

Compaction rules:
1. If the last event for an aggregate (by seq) is DELETED: remove all prior events for that aggregate.
2. If consecutive UPDATED events exist for the same aggregate (no CREATE/DELETE between them):
   keep only the latest, remove older ones.
3. Never compact events with seq >= before_seq (boundary protection).
4. Never delete the last event for a non-deleted aggregate.
"""
from __future__ import annotations

from oprim._logging import log
from oprim.meta_db.duckdb import MetaDB
from oprim.changefeed.errors import ChangefeedCompactorError
from oprim.changefeed.schema import EventType

_DELETED_TYPES = {EventType.SUBSTRATE_DELETED, EventType.DERIVATIVE_DELETED, EventType.NOTE_DELETED}
_UPDATED_TYPES = {EventType.SUBSTRATE_UPDATED, EventType.NOTE_UPDATED}


class ChangefeedCompactor:
    """Compacts old events for a user, removing superseded entries."""

    def __init__(self, db: MetaDB) -> None:
        self.db = db

    def compact_user_events(
        self,
        user_id: str,
        before_seq: int,
        dry_run: bool = False,
    ) -> dict:
        """Compact events with seq < before_seq.

        Returns: {"removed": int, "kept": int, "dry_run": bool}
        """
        rows = self.db.fetchall(
            """SELECT id, event_type, aggregate_id, seq
               FROM changefeed_events
               WHERE user_id = ? AND seq < ?
               ORDER BY seq ASC""",
            [user_id, before_seq],
        )

        to_delete: set[int] = set()

        # Group by aggregate_id
        agg_events: dict[str | None, list[tuple]] = {}
        for row_id, event_type, agg_id, seq in rows:
            agg_events.setdefault(agg_id, []).append((row_id, event_type, seq))

        for agg_id, events in agg_events.items():
            if agg_id is None:
                # Events without aggregate_id are never compacted
                continue

            last_type = EventType(events[-1][1])

            if last_type in _DELETED_TYPES:
                # Aggregate is deleted — all events for it can be removed
                to_delete.update(e[0] for e in events)
                log.info(
                    "compactor_tombstone",
                    agg_id=agg_id,
                    removed=len(events),
                    dry_run=dry_run,
                )
            else:
                # Keep the last event; compact consecutive UPDATED runs
                updated_run: list[int] = []
                for row_id, event_type_str, seq in events:
                    et = EventType(event_type_str)
                    if et in _UPDATED_TYPES:
                        updated_run.append(row_id)
                    else:
                        # Non-update breaks the run; discard all but last UPDATED
                        if len(updated_run) > 1:
                            to_delete.update(updated_run[:-1])
                        updated_run = []
                # Handle trailing UPDATED run (don't delete the very last event)
                if len(updated_run) > 1:
                    to_delete.update(updated_run[:-1])

        removed = len(to_delete)
        kept = len(rows) - removed

        if dry_run:
            log.info("compactor_dry_run", user_id=user_id, would_remove=removed, kept=kept)
            return {"removed": removed, "kept": kept, "dry_run": True}

        if to_delete:
            placeholders = ", ".join("?" * len(to_delete))
            try:
                self.db.execute(
                    f"DELETE FROM changefeed_events WHERE id IN ({placeholders})",
                    list(to_delete),
                )
            except Exception as exc:
                raise ChangefeedCompactorError(f"Compaction DELETE failed: {exc}") from exc

        log.info("compactor_done", user_id=user_id, removed=removed, kept=kept)
        return {"removed": removed, "kept": kept, "dry_run": False}

    def verify_no_gaps(self, user_id: str) -> bool:
        """Verify that seq values are contiguous after compaction (no gaps allowed)."""
        rows = self.db.fetchall(
            "SELECT seq FROM changefeed_events WHERE user_id = ? ORDER BY seq ASC",
            [user_id],
        )
        seqs = [r[0] for r in rows]
        if not seqs:
            return True
        expected_start = seqs[0]
        for i, s in enumerate(seqs):
            if s != expected_start + i:
                log.error("compactor_gap_detected", user_id=user_id, expected=expected_start + i, got=s)
                return False
        return True
