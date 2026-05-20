"""Tests for ChangefeedReader."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from oprim.changefeed.reader import ChangefeedReader, _row_to_event
from oprim.changefeed.schema import EventType
from oprim.changefeed.writer import ChangefeedWriter


@pytest.fixture()
async def populated_db(db):
    """DB with 5 events for user_r and 2 for user_other."""
    w = ChangefeedWriter(db, user_id="user_r", device_id="dev")
    for i, et in enumerate([
        EventType.SUBSTRATE_CREATED,
        EventType.SUBSTRATE_UPDATED,
        EventType.SUBSTRATE_UPDATED,
        EventType.NOTE_CREATED,
        EventType.SUBSTRATE_DELETED,
    ], start=1):
        await w.append(et, f"agg_{i}", {"i": i})

    w2 = ChangefeedWriter(db, user_id="user_other", device_id="dev2")
    await w2.append(EventType.NOTE_CREATED, "n1", {})
    await w2.append(EventType.NOTE_UPDATED, "n1", {})
    return db


class TestReadSince:
    def test_read_all_from_zero(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0)
        assert len(events) == 5

    def test_read_since_seq_is_exclusive(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(2)
        seqs = [e.seq for e in events]
        assert seqs == [3, 4, 5]

    def test_read_since_last_seq_returns_empty(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(5)
        assert events == []

    def test_read_order_is_ascending(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0)
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)

    def test_batch_size_limits_results(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0, batch_size=2)
        assert len(events) == 2

    def test_event_types_filter(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0, event_types=[EventType.SUBSTRATE_UPDATED])
        assert all(e.event_type == EventType.SUBSTRATE_UPDATED for e in events)
        assert len(events) == 2

    def test_multiple_event_types_filter(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0, event_types=[EventType.SUBSTRATE_CREATED, EventType.NOTE_CREATED])
        types = {e.event_type for e in events}
        assert types == {EventType.SUBSTRATE_CREATED, EventType.NOTE_CREATED}

    def test_user_isolation(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0)
        assert all(e.user_id == "user_r" for e in events)

    def test_payload_deserialized(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        events = r.read_since(0)
        assert events[0].payload == {"i": 1}


class TestGetLatestSeq:
    def test_returns_max_seq(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        assert r.get_latest_seq() == 5

    def test_returns_zero_for_empty(self, db):
        r = ChangefeedReader(db, "no_events_user")
        assert r.get_latest_seq() == 0

    def test_independent_per_user(self, populated_db):
        r_a = ChangefeedReader(populated_db, "user_r")
        r_b = ChangefeedReader(populated_db, "user_other")
        assert r_a.get_latest_seq() == 5
        assert r_b.get_latest_seq() == 2


class TestRowToEvent:
    def test_with_string_timestamp(self):
        row = (1, "dev", "user", "substrate.created", "agg1", '{"x":1}', "2024-01-01T10:00:00", 1)
        ev = _row_to_event(row)
        assert ev.created_at.year == 2024
        assert ev.payload == {"x": 1}

    def test_with_datetime_object(self):
        dt = datetime(2024, 6, 1, tzinfo=timezone.utc)
        row = (2, "dev", "user", "note.created", None, '{}', dt, 2)
        ev = _row_to_event(row)
        assert ev.created_at == dt

    def test_with_fallback_timestamp(self):
        row = (3, "dev", "user", "substrate.updated", "a", '{}', None, 3)
        ev = _row_to_event(row)
        assert ev.created_at is not None

    def test_with_dict_payload(self):
        row = (4, "dev", "user", "substrate.created", "b", {"key": "val"}, None, 4)
        ev = _row_to_event(row)
        assert ev.payload == {"key": "val"}


class TestCount:
    def test_count_matches_written(self, populated_db):
        r = ChangefeedReader(populated_db, "user_r")
        assert r.count() == 5

    def test_count_zero_for_new_user(self, db):
        r = ChangefeedReader(db, "nobody")
        assert r.count() == 0
