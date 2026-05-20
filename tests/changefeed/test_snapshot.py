"""Tests for ChangefeedSnapshot (create + restore roundtrip)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from oprim.changefeed.schema import EventType
from oprim.changefeed.snapshot import ChangefeedSnapshot
from oprim.changefeed.writer import ChangefeedWriter
from oprim.storage.protocol import UploadResult


@pytest.fixture()
def mock_storage(tmp_path: Path):
    """A local-filesystem-backed storage mock for tests."""
    store_root = tmp_path / "gdrive_store"
    store_root.mkdir()

    async def _upload(local_path, remote_path, mime_type=None, on_progress=None):
        dest = store_root / remote_path.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local_path).read_bytes())
        return UploadResult(file_id=str(dest), size=dest.stat().st_size, md5="")

    async def _download(file_id, local_path, on_progress=None):
        Path(local_path).write_bytes(Path(file_id).read_bytes())

    svc = MagicMock()
    svc.upload = _upload
    svc.download = _download
    return svc


async def _seed_data(db):
    """Insert substrate, concept and note rows for snapshot testing."""
    for i in range(3):
        db.execute(
            "INSERT INTO substrate (id, ulid, title, meta_json) VALUES (?, ?, ?, '{}')",
            [f"sub_{i}", f"ulid_{i}", f"Title {i}"],
        )
    db.execute(
        "INSERT INTO concept (id, name, meta_json) VALUES (?, ?, '{}')",
        ["c1", "ConceptA"],
    )
    db.execute(
        "INSERT INTO note (id, title, content, meta_json) VALUES (?, ?, ?, '{}')",
        ["n1", "Note1", "body"],
    )


# Keep old alias for tests that only need substrates
async def _seed_substrates(db):
    await _seed_data(db)


class TestCreateSnapshot:
    async def test_create_uploads_file(self, db, mock_storage):
        await _seed_data(db)
        snap = ChangefeedSnapshot(db)
        result = await snap.create_snapshot("u1", "dev_a", mock_storage)
        assert result["file_id"]
        assert result["substrate_count"] == 3
        assert result["concept_count"] == 1
        assert result["note_count"] == 1
        assert result["seq_at"] == 0

    async def test_create_records_in_db(self, db, mock_storage):
        snap = ChangefeedSnapshot(db)
        result = await snap.create_snapshot("u1", "dev_a", mock_storage)
        snaps = snap.list_snapshots("u1")
        assert len(snaps) == 1
        assert snaps[0]["snapshot_id"] == result["snapshot_id"]

    async def test_snapshot_seq_at_reflects_latest(self, db, mock_storage):
        w = ChangefeedWriter(db, user_id="u1", device_id="dev")
        await w.append(EventType.SUBSTRATE_CREATED, "s1", {})
        await w.append(EventType.SUBSTRATE_UPDATED, "s1", {})
        snap = ChangefeedSnapshot(db)
        result = await snap.create_snapshot("u1", "dev", mock_storage)
        assert result["seq_at"] == 2

    async def test_create_twice_creates_two_records(self, db, mock_storage):
        snap = ChangefeedSnapshot(db)
        await snap.create_snapshot("u1", "dev", mock_storage)
        await snap.create_snapshot("u1", "dev", mock_storage)
        snaps = snap.list_snapshots("u1")
        assert len(snaps) == 2


class TestRestoreFromSnapshot:
    async def test_restore_repopulates_substrate(self, db, mock_storage):
        await _seed_substrates(db)
        snap = ChangefeedSnapshot(db)
        result = await snap.create_snapshot("u1", "dev", mock_storage)
        file_id = result["file_id"]

        # Delete all substrates
        db.execute("DELETE FROM substrate")
        assert db.fetchall("SELECT COUNT(*) FROM substrate")[0][0] == 0

        # Restore
        restore_result = await snap.restore_from_snapshot(file_id, mock_storage)
        assert restore_result["substrate_count"] == 3
        assert db.fetchall("SELECT COUNT(*) FROM substrate")[0][0] == 3

    async def test_restore_is_equivalent_to_original(self, db, mock_storage):
        await _seed_substrates(db)
        snap = ChangefeedSnapshot(db)
        result = await snap.create_snapshot("u1", "dev", mock_storage)

        # Change data then restore
        db.execute("DELETE FROM substrate")
        db.execute("INSERT INTO substrate (id, ulid, title, meta_json) VALUES ('x', 'x', 'Different', '{}')")

        await snap.restore_from_snapshot(result["file_id"], mock_storage)
        rows = db.fetchall("SELECT id FROM substrate ORDER BY id")
        ids = [r[0] for r in rows]
        assert "sub_0" in ids
        assert "x" not in ids

    async def test_restore_returns_seq_at(self, db, mock_storage):
        w = ChangefeedWriter(db, user_id="u1", device_id="d")
        await w.append(EventType.SUBSTRATE_CREATED, "s1", {})
        snap = ChangefeedSnapshot(db)
        cr = await snap.create_snapshot("u1", "d", mock_storage)

        db.execute("DELETE FROM substrate")
        rr = await snap.restore_from_snapshot(cr["file_id"], mock_storage)
        assert rr["seq_at"] == 1

    async def test_restore_invalid_format_raises(self, db, tmp_path, mock_storage):
        import json
        bad_file = tmp_path / "bad.json"
        bad_file.write_text(json.dumps({"format": "wrong_format", "substrate": []}))

        snap = ChangefeedSnapshot(db)

        async def _dl(fid, lp, on_progress=None):
            Path(lp).write_bytes(bad_file.read_bytes())

        bad_storage = MagicMock()
        bad_storage.download = _dl

        with pytest.raises(ValueError, match="Unknown snapshot format"):
            await snap.restore_from_snapshot(str(bad_file), bad_storage)


class TestListSnapshots:
    async def test_empty_list_for_new_user(self, db):
        snap = ChangefeedSnapshot(db)
        assert snap.list_snapshots("nobody") == []

    async def test_ordered_newest_first(self, db, mock_storage):
        snap = ChangefeedSnapshot(db)
        w = ChangefeedWriter(db, user_id="u1", device_id="d")
        await snap.create_snapshot("u1", "d", mock_storage)
        await w.append(EventType.SUBSTRATE_CREATED, "s1", {})
        await snap.create_snapshot("u1", "d", mock_storage)
        snaps = snap.list_snapshots("u1")
        assert snaps[0]["seq_at"] >= snaps[1]["seq_at"]
