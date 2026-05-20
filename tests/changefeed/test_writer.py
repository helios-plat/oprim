"""Tests for ChangefeedWriter."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

from oprim.changefeed.errors import ChangefeedConflictError
from oprim.changefeed.schema import ChangefeedEvent, EventType
from oprim.changefeed.writer import ChangefeedWriter


class TestAppend:
    async def test_append_returns_event(self, db):
        w = ChangefeedWriter(db, user_id="u1", device_id="dev_a")
        ev = await w.append(EventType.SUBSTRATE_CREATED, "agg_1", {"title": "t"})
        assert isinstance(ev, ChangefeedEvent)
        assert ev.event_type == EventType.SUBSTRATE_CREATED
        assert ev.user_id == "u1"
        assert ev.device_id == "dev_a"
        assert ev.aggregate_id == "agg_1"
        assert ev.payload == {"title": "t"}
        assert ev.seq == 1

    async def test_seq_increments(self, db):
        w = ChangefeedWriter(db, user_id="u1", device_id="dev_a")
        ev1 = await w.append(EventType.SUBSTRATE_CREATED, "agg_1", {})
        ev2 = await w.append(EventType.SUBSTRATE_UPDATED, "agg_1", {"title": "new"})
        ev3 = await w.append(EventType.NOTE_CREATED, "note_1", {})
        assert ev1.seq == 1
        assert ev2.seq == 2
        assert ev3.seq == 3

    async def test_seq_per_user_independent(self, db):
        w_a = ChangefeedWriter(db, user_id="user_a", device_id="dev_a")
        w_b = ChangefeedWriter(db, user_id="user_b", device_id="dev_b")
        ev_a1 = await w_a.append(EventType.SUBSTRATE_CREATED, "a", {})
        ev_a2 = await w_a.append(EventType.SUBSTRATE_UPDATED, "a", {})
        ev_b1 = await w_b.append(EventType.SUBSTRATE_CREATED, "b", {})
        assert ev_a1.seq == 1
        assert ev_a2.seq == 2
        assert ev_b1.seq == 1  # independent counter for user_b

    async def test_payload_serialized(self, db):
        w = ChangefeedWriter(db, user_id="u1", device_id="d")
        payload = {"title": "测试", "tags": ["a", "b"], "count": 42}
        ev = await w.append(EventType.NOTE_CREATED, "n1", payload)
        assert ev.payload == payload

    async def test_aggregate_id_optional(self, db):
        w = ChangefeedWriter(db, user_id="u1", device_id="d")
        ev = await w.append(EventType.CONCEPT_CREATED, None, {"name": "test"})
        assert ev.aggregate_id is None

    async def test_event_stored_in_db(self, db):
        w = ChangefeedWriter(db, user_id="u1", device_id="d")
        await w.append(EventType.SUBSTRATE_CREATED, "agg_1", {"x": 1})
        rows = db.fetchall("SELECT COUNT(*) FROM changefeed_events WHERE user_id = 'u1'")
        assert rows[0][0] == 1

    async def test_concurrent_writers_no_seq_collision(self, db):
        """Two coroutines writing concurrently must produce unique seqs."""
        w = ChangefeedWriter(db, user_id="u_concurrent", device_id="dev")
        results = await asyncio.gather(
            *[w.append(EventType.SUBSTRATE_UPDATED, f"agg_{i}", {"i": i}) for i in range(10)]
        )
        seqs = sorted(ev.seq for ev in results)
        assert seqs == list(range(1, 11)), f"Expected 1..10, got {seqs}"

    async def test_unique_constraint_raises_conflict_error(self, db):
        """Cover the except/ChangefeedConflictError path (lines 69-74)."""
        w = ChangefeedWriter(db, user_id="u_conflict", device_id="dev")
        import duckdb as _duckdb
        original_execute = db.execute

        call_count = [0]
        def _patched_execute(sql, params=None):
            if "INSERT INTO changefeed_events" in sql:
                call_count[0] += 1
                if call_count[0] == 1:
                    err = _duckdb.ConstraintException("UNIQUE constraint violated")
                    raise err
            return original_execute(sql, params)

        with patch.object(db, "execute", side_effect=_patched_execute):
            with pytest.raises(ChangefeedConflictError):
                await w.append(EventType.SUBSTRATE_CREATED, "agg", {})

    async def test_non_constraint_exception_reraises(self, db):
        """Non-constraint exceptions propagate as-is."""
        w = ChangefeedWriter(db, user_id="u_err", device_id="dev")
        from oprim.errors import MetaDBError

        def _patched(sql, params=None):
            if "INSERT INTO changefeed_events" in sql:
                raise MetaDBError("disk full")
            return db.__class__.execute(db, sql, params)

        with patch.object(db, "execute", side_effect=_patched):
            with pytest.raises(MetaDBError, match="disk full"):
                await w.append(EventType.SUBSTRATE_CREATED, "agg", {})

    async def test_created_at_datetime_branch(self, db):
        """When DB returns datetime object, it flows through elif branch."""
        w = ChangefeedWriter(db, user_id="u_dt", device_id="dev")
        now = datetime(2024, 6, 1, tzinfo=timezone.utc)

        original_fetchall = db.fetchall
        call_count = [0]

        def _patched_fetchall(sql, params=None):
            rows = original_fetchall(sql, params)
            call_count[0] += 1
            # On the second fetchall (SELECT id, created_at), replace timestamp with datetime obj
            if "created_at" in sql and call_count[0] == 2 and rows:
                return [(rows[0][0], now)]
            return rows

        with patch.object(db, "fetchall", side_effect=_patched_fetchall):
            ev = await w.append(EventType.NOTE_CREATED, "n1", {})

        assert ev.created_at == now

    async def test_two_devices_independent_seqs(self, db):
        """Two devices writing for same user share a seq counter."""
        w_a = ChangefeedWriter(db, user_id="shared_user", device_id="device_a")
        w_b = ChangefeedWriter(db, user_id="shared_user", device_id="device_b")
        ev1 = await w_a.append(EventType.SUBSTRATE_CREATED, "a", {})
        ev2 = await w_b.append(EventType.SUBSTRATE_CREATED, "b", {})
        ev3 = await w_a.append(EventType.SUBSTRATE_UPDATED, "a", {})
        assert ev1.seq == 1
        assert ev2.seq == 2
        assert ev3.seq == 3
