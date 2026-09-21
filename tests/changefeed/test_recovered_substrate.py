"""Tests for the recovered oprim sync substrate (changefeed + storage protocol).

Validates the canonical contract required by oskill.sync consumers:
- storage protocol dataclasses + StorageAdapter Protocol
- ChangefeedEvent field contract (event_type / seq / device_id) + dotted EventType
- append-only writer / cursor reader with lazy schema bootstrap
- snapshot create/restore round-trip against plural stratum tables
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oprim.changefeed import (
    ChangefeedEvent,
    ChangefeedReader,
    ChangefeedSnapshot,
    ChangefeedWriter,
    EventType,
)
from oprim.meta_db.duckdb import open_meta_db
from oprim.storage.protocol import (
    StorageAdapter,
    StorageFile,
    StorageQuota,
    UploadResult,
)

_STRATUM_DDL = """
CREATE TABLE IF NOT EXISTS substrates (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT, mime TEXT,
    source_path TEXT, file_hash TEXT, byte_size BIGINT, page_count INTEGER,
    parser TEXT, language TEXT, has_cjk BOOLEAN DEFAULT FALSE,
    is_scanned BOOLEAN DEFAULT FALSE, is_pinned BOOLEAN DEFAULT FALSE,
    pinned_at TEXT, pin_priority INTEGER DEFAULT 0, created_at TEXT,
    updated_at TEXT, meta_json TEXT DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS concepts (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'concept_idea', aliases TEXT[], wikilink TEXT,
    substrate_refs TEXT[], related_concept_ids TEXT[], created_at TEXT,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY, title TEXT, content TEXT, wikilinks TEXT DEFAULT '[]',
    substrate_id TEXT, meta_json TEXT DEFAULT '{}', created_at TEXT,
    updated_at TEXT
);
"""


@pytest.fixture()
def db(tmp_path: Path):
    m = open_meta_db(tmp_path / "meta.duckdb")
    for stmt in _STRATUM_DDL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            m.execute(stmt)
    yield m
    m.close()


class _FakeStorage:
    def __init__(self) -> None:
        self.uploaded: dict[str, str] = {}

    async def upload(self, local_path, remote_path, mime_type=None):
        self.uploaded[remote_path] = Path(local_path).read_text()
        return UploadResult(file_id="file_1", size=len(self.uploaded[remote_path]), md5="m")

    async def download(self, file_id, local_path, on_progress=None):
        Path(local_path).write_bytes(next(iter(self.uploaded.values())).encode())


# ── storage protocol ─────────────────────────────────────────────────────────


def test_storage_dataclasses_and_protocol():
    uf = StorageFile(
        file_id="f1",
        name="a.json",
        size=3,
        mime_type="application/json",
        created_at=None,
        modified_at=None,
        md5=None,
    )
    assert uf.file_id == "f1" and uf.name == "a.json"
    assert UploadResult(file_id="f1", size=3, md5="x").size == 3
    assert StorageQuota(used_bytes=1, total_bytes=2, available_bytes=1).available_bytes == 1
    assert getattr(StorageAdapter, "_is_runtime_protocol", False) is True


# ── changefeed event contract ────────────────────────────────────────────────


def test_event_contract_is_dotted_and_single_authority():
    assert EventType.SUBSTRATE_CREATED.value == "substrate.created"
    ev = ChangefeedEvent(
        id=1,
        device_id="d1",
        user_id="u1",
        event_type=EventType.NOTE_CREATED,
        aggregate_id="n1",
        payload={"x": 1},
        created_at=__import__("datetime").datetime.now(),
        seq=7,
    )
    d = ev.to_dict()
    assert set(d) == {
        "id",
        "device_id",
        "user_id",
        "event_type",
        "aggregate_id",
        "payload",
        "created_at",
        "seq",
    }
    assert d["event_type"] == "note.created" and d["seq"] == 7 and d["device_id"] == "d1"
    round_trip = ChangefeedEvent.from_dict(d)
    assert round_trip.event_type is EventType.NOTE_CREATED and round_trip.seq == 7


# ── writer / reader (append-only + lazy schema) ──────────────────────────────


@pytest.mark.asyncio()
async def test_writer_seq_monotonic_and_reader_cursor(db):
    writer = ChangefeedWriter(db, user_id="u1", device_id="dev_a")
    e1 = await writer.append(EventType.SUBSTRATE_CREATED, "sub_1", {"title": "A"})
    e2 = await writer.append(EventType.SUBSTRATE_UPDATED, "sub_1", {"title": "B"})
    e3 = await writer.append(EventType.NOTE_CREATED, "note_1", {"title": "N"})
    assert [e1.seq, e2.seq, e3.seq] == [1, 2, 3]
    assert e1.device_id == "dev_a" and e1.user_id == "u1"

    reader = ChangefeedReader(db, "u1")
    assert reader.get_latest_seq() == 3
    assert reader.count() == 3
    assert [e.seq for e in reader.read_since(1)] == [2, 3]
    assert [e.event_type for e in reader.read_since(0, event_types=[EventType.NOTE_CREATED])] == [
        EventType.NOTE_CREATED
    ]


@pytest.mark.asyncio()
async def test_writer_seq_is_per_user(db):
    a = ChangefeedWriter(db, user_id="ua", device_id="d")
    b = ChangefeedWriter(db, user_id="ub", device_id="d")
    assert (await a.append(EventType.SUBSTRATE_CREATED, "x", {})).seq == 1
    assert (await a.append(EventType.SUBSTRATE_CREATED, "x", {})).seq == 2
    assert (await b.append(EventType.SUBSTRATE_CREATED, "y", {})).seq == 1


# ── snapshot round-trip ──────────────────────────────────────────────────────


@pytest.mark.asyncio()
async def test_snapshot_backup_restore_round_trip(db):
    writer = ChangefeedWriter(db, user_id="u1", device_id="d1")
    await writer.append(EventType.SUBSTRATE_CREATED, "sub_1", {"title": "A"})
    db.execute(
        "INSERT INTO substrates (id, user_id, title, meta_json) VALUES (?, ?, ?, ?)",
        ["sub_1", "u1", "Doc A", "{}"],
    )
    db.execute(
        "INSERT INTO notes (id, title, content, meta_json) VALUES (?, ?, ?, ?)",
        ["note_1", "N", "c", "{}"],
    )

    storage = _FakeStorage()
    snap = ChangefeedSnapshot(db)
    out = await snap.create_snapshot("u1", "d1", storage)
    assert out["substrate_count"] == 1 and out["note_count"] == 1
    assert out["seq_at"] == 1 and out["file_id"] == "file_1"

    rows = db.fetchall("SELECT id FROM changefeed_snapshots WHERE user_id = ?", ["u1"])
    assert len(rows) == 1 and rows[0][0] == out["snapshot_id"]

    restored = await snap.restore_from_snapshot("file_1", storage)
    assert restored["snapshot_id"] == out["snapshot_id"]
    assert restored["substrate_count"] == 1
    got = db.fetchall("SELECT title FROM substrates WHERE id = 'sub_1'")
    assert got[0][0] == "Doc A"


@pytest.mark.asyncio()
async def test_restore_rejects_unknown_format(db):
    storage = _FakeStorage()
    storage.uploaded["x"] = json.dumps({"format": "unknown_v99"})
    with pytest.raises(ValueError, match="snapshot format"):
        await ChangefeedSnapshot(db).restore_from_snapshot("file_1", storage)
