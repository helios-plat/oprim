"""Changefeed snapshot — serialize full DB state to JSON for cold-start sync."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ulid import ULID

from oprim._logging import log
from oprim.meta_db.duckdb import MetaDB
from oprim.changefeed.reader import ChangefeedReader

_FORMAT = "stratum_snapshot_v1"


class ChangefeedSnapshot:
    """Creates and restores full-state snapshots for cross-device cold start."""

    def __init__(self, db: MetaDB) -> None:
        self.db = db

    async def create_snapshot(
        self,
        user_id: str,
        device_id: str,
        storage_adapter,
    ) -> dict:
        """Serialize current DB state, upload to storage, record in changefeed_snapshots."""
        reader = ChangefeedReader(self.db, user_id)
        seq_at = reader.get_latest_seq()

        snapshot_id = str(ULID())
        created_at = datetime.now(tz=timezone.utc)

        # Gather tables
        substrates = self.db.fetchall(
            "SELECT id, ulid, title, mime, source_path, file_hash, byte_size, page_count, "
            "parser, language, has_cjk, is_scanned, created_at, updated_at, meta_json "
            "FROM substrate"
        )
        concepts = self.db.fetchall(
            "SELECT id, name, aliases, description, wikilink, source_ids, meta_json, "
            "created_at, updated_at FROM concept"
        )
        notes = self.db.fetchall(
            "SELECT id, title, content, wikilinks, substrate_id, meta_json, "
            "created_at, updated_at FROM note"
        )

        data = {
            "format": _FORMAT,
            "snapshot_id": snapshot_id,
            "user_id": user_id,
            "device_id": device_id,
            "seq_at": seq_at,
            "created_at": created_at.isoformat(),
            "substrate": [_row_to_dict(r, _SUBSTRATE_COLS) for r in substrates],
            "concept": [_row_to_dict(r, _CONCEPT_COLS) for r in concepts],
            "note": [_row_to_dict(r, _NOTE_COLS) for r in notes],
        }

        # Write to temp file, upload, cleanup
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
            json.dump(data, tf, ensure_ascii=False, default=str)
            tmp_path = tf.name

        remote_path = f"snapshots/snapshot_{user_id}_{seq_at}.json"
        try:
            result = await storage_adapter.upload(tmp_path, remote_path, mime_type="application/json")
            file_id = result.file_id
        finally:
            os.unlink(tmp_path)

        # Record snapshot in DB
        self.db.execute(
            "INSERT INTO changefeed_snapshots (id, user_id, device_id, seq_at, file_id) VALUES (?, ?, ?, ?, ?)",
            [snapshot_id, user_id, device_id, seq_at, file_id],
        )

        log.info(
            "snapshot_created",
            snapshot_id=snapshot_id,
            seq_at=seq_at,
            file_id=file_id,
            user_id=user_id,
        )
        return {
            "snapshot_id": snapshot_id,
            "seq_at": seq_at,
            "file_id": file_id,
            "substrate_count": len(substrates),
            "concept_count": len(concepts),
            "note_count": len(notes),
        }

    async def restore_from_snapshot(
        self,
        snapshot_file_id: str,
        storage_adapter,
    ) -> dict:
        """Download snapshot and repopulate local DB tables.

        WARNING: This truncates substrate, concept, note tables before restoring.
        Should only be called on a fresh install or with explicit user consent.
        """
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tf:
            tmp_path = tf.name

        try:
            await storage_adapter.download(snapshot_file_id, tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
        finally:
            os.unlink(tmp_path)

        if data.get("format") != _FORMAT:
            raise ValueError(f"Unknown snapshot format: {data.get('format')!r}")

        # Truncate + restore
        for table in ("substrate", "concept", "note"):
            self.db.execute(f"DELETE FROM {table}")

        for row in data.get("substrate", []):
            self.db.execute(
                "INSERT INTO substrate (id, ulid, title, mime, source_path, file_hash, "
                "byte_size, page_count, parser, language, has_cjk, is_scanned, "
                "created_at, updated_at, meta_json) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _SUBSTRATE_COLS],
            )

        for row in data.get("concept", []):
            self.db.execute(
                "INSERT INTO concept (id, name, aliases, description, wikilink, source_ids, "
                "meta_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _CONCEPT_COLS],
            )

        for row in data.get("note", []):
            self.db.execute(
                "INSERT INTO note (id, title, content, wikilinks, substrate_id, meta_json, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [row.get(c) for c in _NOTE_COLS],
            )

        seq_at = data.get("seq_at", 0)
        log.info(
            "snapshot_restored",
            snapshot_file_id=snapshot_file_id,
            seq_at=seq_at,
            substrate_count=len(data.get("substrate", [])),
        )
        return {
            "seq_at": seq_at,
            "snapshot_id": data.get("snapshot_id"),
            "substrate_count": len(data.get("substrate", [])),
            "concept_count": len(data.get("concept", [])),
            "note_count": len(data.get("note", [])),
        }

    def list_snapshots(self, user_id: str) -> list[dict]:
        """Return all snapshots for user, newest first."""
        rows = self.db.fetchall(
            "SELECT id, user_id, device_id, seq_at, file_id, created_at "
            "FROM changefeed_snapshots WHERE user_id = ? ORDER BY seq_at DESC",
            [user_id],
        )
        return [
            {
                "snapshot_id": r[0],
                "user_id": r[1],
                "device_id": r[2],
                "seq_at": r[3],
                "file_id": r[4],
                "created_at": str(r[5]),
            }
            for r in rows
        ]


# ── column lists ──────────────────────────────────────────────────────────────

_SUBSTRATE_COLS = [
    "id", "ulid", "title", "mime", "source_path", "file_hash",
    "byte_size", "page_count", "parser", "language", "has_cjk",
    "is_scanned", "created_at", "updated_at", "meta_json",
]
_CONCEPT_COLS = ["id", "name", "aliases", "description", "wikilink", "source_ids", "meta_json", "created_at", "updated_at"]
_NOTE_COLS = ["id", "title", "content", "wikilinks", "substrate_id", "meta_json", "created_at", "updated_at"]


def _row_to_dict(row: tuple, cols: list[str]) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in zip(cols, row)}
