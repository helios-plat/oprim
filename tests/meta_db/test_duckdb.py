"""Integration tests for oprim.meta_db.duckdb (uses tmp_path)."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.meta_db.duckdb import MetaDB, open_meta_db
from oprim.errors import MetaDBError

_MIGRATIONS_DIR = (
    Path(__file__).parent.parent.parent
    / "oprim" / "meta_db" / "migrations"
)


class TestMetaDB:
    def test_open_creates_file(self, tmp_path: Path):
        db_path = tmp_path / "meta.duckdb"
        db = open_meta_db(db_path)
        db.connect()
        db.close()
        assert db_path.exists()

    def test_execute_and_fetchall(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.execute("CREATE TABLE t (x INTEGER, y TEXT)")
        db.execute("INSERT INTO t VALUES (?, ?)", [42, "hello"])
        rows = db.fetchall("SELECT x, y FROM t")
        assert rows == [(42, "hello")]
        db.close()

    def test_multiple_rows(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        for i in range(5):
            db.execute("INSERT INTO items VALUES (?, ?)", [i, f"item_{i}"])
        rows = db.fetchall("SELECT COUNT(*) FROM items")
        assert rows[0][0] == 5
        db.close()

    def test_migrate_applies_initial_schema(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.migrate(_MIGRATIONS_DIR)
        # Check that substrate table exists
        rows = db.fetchall(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'substrate'"
        )
        assert len(rows) == 1
        db.close()

    def test_migrate_idempotent(self, tmp_path: Path):
        """Running migrations twice should not fail."""
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.migrate(_MIGRATIONS_DIR)
        db.migrate(_MIGRATIONS_DIR)  # must not raise
        db.close()

    def test_migrate_tracking(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.migrate(_MIGRATIONS_DIR)
        rows = db.fetchall("SELECT filename FROM _migrations")
        filenames = [r[0] for r in rows]
        assert "001_initial.sql" in filenames
        db.close()

    def test_substrate_table_crud(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.migrate(_MIGRATIONS_DIR)
        # Insert
        db.execute(
            "INSERT INTO substrate (id, ulid, title, mime) VALUES (?, ?, ?, ?)",
            ["sub_1", "01JTEST", "Test Doc", "application/pdf"],
        )
        rows = db.fetchall("SELECT id, title FROM substrate WHERE id = ?", ["sub_1"])
        assert rows[0] == ("sub_1", "Test Doc")
        # Update
        db.execute("UPDATE substrate SET title = ? WHERE id = ?", ["Updated", "sub_1"])
        rows = db.fetchall("SELECT title FROM substrate WHERE id = ?", ["sub_1"])
        assert rows[0][0] == "Updated"
        # Delete
        db.execute("DELETE FROM substrate WHERE id = ?", ["sub_1"])
        rows = db.fetchall("SELECT id FROM substrate WHERE id = ?", ["sub_1"])
        assert rows == []
        db.close()

    def test_close_is_idempotent(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        db.connect()
        db.close()
        db.close()  # second close must not raise

    def test_execute_bad_sql_raises(self, tmp_path: Path):
        db = open_meta_db(tmp_path / "meta.duckdb")
        with pytest.raises(MetaDBError):
            db.execute("SELECT * FROM nonexistent_table_xyz")
        db.close()
