"""Tests for ChangefeedCompactor."""
from __future__ import annotations

import pytest

from oprim.changefeed.compactor import ChangefeedCompactor
from oprim.changefeed.reader import ChangefeedReader
from oprim.changefeed.schema import EventType
from oprim.changefeed.writer import ChangefeedWriter


async def _write_events(db, user_id, events):
    """Helper: write list of (event_type, aggregate_id) tuples."""
    w = ChangefeedWriter(db, user_id=user_id, device_id="dev")
    for et, agg_id in events:
        await w.append(et, agg_id, {"step": et.value})


class TestCompactorTombstone:
    async def test_deleted_aggregate_removes_all_events(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
            (EventType.SUBSTRATE_DELETED, "s1"),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 3

        r = ChangefeedReader(db, "u")
        assert r.count() == 0

    async def test_non_deleted_aggregate_kept(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 0

        r = ChangefeedReader(db, "u")
        assert r.count() == 2

    async def test_mixed_deleted_and_active(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_DELETED, "s1"),  # s1 deleted
            (EventType.SUBSTRATE_CREATED, "s2"),
            (EventType.SUBSTRATE_UPDATED, "s2"),  # s2 still active
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 2  # s1's 2 events gone

        r = ChangefeedReader(db, "u")
        remaining = r.read_since(0)
        agg_ids = {e.aggregate_id for e in remaining}
        assert "s1" not in agg_ids
        assert "s2" in agg_ids


class TestCompactorConsecutiveUpdates:
    async def test_consecutive_updates_keeps_latest(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 2  # 3 UPDATEDs → keep 1

        r = ChangefeedReader(db, "u")
        events = r.read_since(0)
        event_types = [e.event_type for e in events]
        assert event_types.count(EventType.SUBSTRATE_UPDATED) == 1

    async def test_single_update_not_removed(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 0


class TestCompactorBoundary:
    async def test_before_seq_boundary_respected(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),  # seq 1
            (EventType.SUBSTRATE_DELETED, "s1"),  # seq 2
            (EventType.SUBSTRATE_CREATED, "s2"),  # seq 3
        ])
        c = ChangefeedCompactor(db)
        # Only compact up to seq 2 — s1 events (seq 1, 2) are included
        # seq 3 is >= before_seq=3 so not touched
        result = c.compact_user_events("u", before_seq=3)
        assert result["removed"] == 2  # s1's create + delete

        r = ChangefeedReader(db, "u")
        events = r.read_since(0)
        assert len(events) == 1  # only s2 create remains
        assert events[0].aggregate_id == "s2"


class TestCompactorDryRun:
    async def test_dry_run_does_not_modify(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_DELETED, "s1"),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100, dry_run=True)
        assert result["dry_run"] is True
        assert result["removed"] == 2  # would remove 2

        r = ChangefeedReader(db, "u")
        assert r.count() == 2  # still 2 — dry_run didn't delete


class TestCompactorNullAgg:
    async def test_null_aggregate_id_never_compacted(self, db):
        await _write_events(db, "u", [
            (EventType.CONCEPT_CREATED, None),
            (EventType.CONCEPT_LINKED, None),
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u", before_seq=100)
        assert result["removed"] == 0


class TestCompactorInterruptedRun:
    async def test_update_run_interrupted_by_non_update(self, db):
        """UPDATE, UPDATE, CONCEPT_LINKED, UPDATE → first two UPDATEs compacted (run broken by non-update)."""
        await _write_events(db, "u_int", [
            (EventType.SUBSTRATE_CREATED, "s1"),   # seq 1
            (EventType.SUBSTRATE_UPDATED, "s1"),   # seq 2
            (EventType.SUBSTRATE_UPDATED, "s1"),   # seq 3 — run of 2 UPDATEDs interrupted below
            (EventType.CONCEPT_LINKED, "s1"),      # seq 4 — breaks the UPDATED run
            (EventType.SUBSTRATE_UPDATED, "s1"),   # seq 5
        ])
        c = ChangefeedCompactor(db)
        result = c.compact_user_events("u_int", before_seq=100)
        # The two consecutive UPDATEs (seq 2, 3) before CONCEPT_LINKED → keep seq 3, remove seq 2
        assert result["removed"] >= 1  # at least seq 2 removed


class TestCompactorDeleteError:
    async def test_delete_failure_raises_compactor_error(self, db):
        from unittest.mock import patch
        from oprim.errors import MetaDBError
        from oprim.changefeed.errors import ChangefeedCompactorError

        await _write_events(db, "u_del_err", [
            (EventType.SUBSTRATE_CREATED, "s_del"),
            (EventType.SUBSTRATE_DELETED, "s_del"),
        ])
        c = ChangefeedCompactor(db)
        original_execute = db.execute

        def _fail_on_delete(sql, params=None):
            if "DELETE FROM changefeed_events WHERE id" in sql:
                raise MetaDBError("simulated delete failure")
            return original_execute(sql, params)

        with patch.object(db, "execute", side_effect=_fail_on_delete):
            with pytest.raises(ChangefeedCompactorError, match="Compaction DELETE failed"):
                c.compact_user_events("u_del_err", before_seq=100)


class TestVerifyNoGaps:
    async def test_no_gaps_after_write(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
            (EventType.NOTE_CREATED, "n1"),
        ])
        c = ChangefeedCompactor(db)
        assert c.verify_no_gaps("u") is True

    def test_empty_user_has_no_gaps(self, db):
        c = ChangefeedCompactor(db)
        assert c.verify_no_gaps("nobody") is True

    async def test_gaps_detected_after_manual_delete(self, db):
        await _write_events(db, "u", [
            (EventType.SUBSTRATE_CREATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
            (EventType.SUBSTRATE_UPDATED, "s1"),
        ])
        # Manually delete seq=2 to create a gap
        db.execute("DELETE FROM changefeed_events WHERE user_id = 'u' AND seq = 2")
        c = ChangefeedCompactor(db)
        assert c.verify_no_gaps("u") is False
