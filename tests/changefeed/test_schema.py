"""Tests for changefeed schema."""
from __future__ import annotations

from datetime import datetime, timezone

from oprim.changefeed.schema import ChangefeedEvent, EventType


class TestEventType:
    def test_all_values_are_strings(self):
        for et in EventType:
            assert isinstance(et.value, str)
            assert "." in et.value

    def test_substrate_events_exist(self):
        assert EventType.SUBSTRATE_CREATED.value == "substrate.created"
        assert EventType.SUBSTRATE_UPDATED.value == "substrate.updated"
        assert EventType.SUBSTRATE_DELETED.value == "substrate.deleted"

    def test_phase15_events_exist(self):
        assert EventType.SUBSTRATE_PINNED
        assert EventType.SUBSTRATE_UNPINNED

    def test_from_string(self):
        et = EventType("substrate.created")
        assert et == EventType.SUBSTRATE_CREATED


class TestChangefeedEvent:
    def _make(self, seq=1, event_type=EventType.SUBSTRATE_CREATED):
        return ChangefeedEvent(
            id=1,
            device_id="dev_a",
            user_id="user_123",
            event_type=event_type,
            aggregate_id="agg_ulid_1",
            payload={"title": "test"},
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            seq=seq,
        )

    def test_to_dict_roundtrip(self):
        ev = self._make()
        d = ev.to_dict()
        ev2 = ChangefeedEvent.from_dict(d)
        assert ev2.id == ev.id
        assert ev2.event_type == ev.event_type
        assert ev2.seq == ev.seq
        assert ev2.payload == ev.payload

    def test_to_dict_event_type_is_string(self):
        ev = self._make()
        d = ev.to_dict()
        assert isinstance(d["event_type"], str)

    def test_aggregate_id_optional(self):
        ev = ChangefeedEvent(
            id=1, device_id="d", user_id="u",
            event_type=EventType.NOTE_CREATED,
            aggregate_id=None,
            payload={},
            created_at=datetime.now(tz=timezone.utc),
            seq=1,
        )
        d = ev.to_dict()
        assert d["aggregate_id"] is None
        ev2 = ChangefeedEvent.from_dict(d)
        assert ev2.aggregate_id is None
