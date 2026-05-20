"""Changefeed event schema — EventType enum + ChangefeedEvent dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    SUBSTRATE_CREATED = "substrate.created"
    SUBSTRATE_UPDATED = "substrate.updated"
    SUBSTRATE_DELETED = "substrate.deleted"
    SUBSTRATE_PINNED = "substrate.pinned"
    SUBSTRATE_UNPINNED = "substrate.unpinned"
    DERIVATIVE_CREATED = "derivative.created"
    DERIVATIVE_DELETED = "derivative.deleted"
    NOTE_CREATED = "note.created"
    NOTE_UPDATED = "note.updated"
    NOTE_DELETED = "note.deleted"
    CONCEPT_CREATED = "concept.created"
    CONCEPT_LINKED = "concept.linked"
    CONCEPT_UNLINKED = "concept.unlinked"


@dataclass
class ChangefeedEvent:
    id: int
    device_id: str
    user_id: str
    event_type: EventType
    aggregate_id: str | None
    payload: dict
    created_at: datetime
    seq: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "device_id": self.device_id,
            "user_id": self.user_id,
            "event_type": self.event_type.value,
            "aggregate_id": self.aggregate_id,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChangefeedEvent":
        return cls(
            id=d["id"],
            device_id=d["device_id"],
            user_id=d["user_id"],
            event_type=EventType(d["event_type"]),
            aggregate_id=d.get("aggregate_id"),
            payload=d["payload"],
            created_at=datetime.fromisoformat(d["created_at"]),
            seq=d["seq"],
        )
