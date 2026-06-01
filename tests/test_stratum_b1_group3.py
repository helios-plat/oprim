"""Tests for Stratum Batch 1 Group 3: db_insert, db_query, db_write, db_read,
db_soft_delete, db_update, migration_runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg
import pytest

from oprim._db_types import AccessResult, WriteResult
from oprim._exceptions import OprimError
from oprim.db_insert import db_insert
from oprim.db_query import db_query
from oprim.db_read import db_read
from oprim.db_soft_delete import db_soft_delete
from oprim.db_update import db_update
from oprim.db_write import db_write
from oprim.migration_runner import MigrationResult, migration_runner

DSN = "postgresql://test/testdb"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cursor(fetchone=None, fetchmany=None, rowcount=1):
    """Build a mock cursor context manager."""
    cur = MagicMock()
    cur.fetchone.return_value = fetchone
    cur.fetchmany.return_value = fetchmany if fetchmany is not None else []
    cur.rowcount = rowcount
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


def _make_conn(cursor):
    """Build a mock connection context manager."""
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    return conn


# ---------------------------------------------------------------------------
# db_insert
# ---------------------------------------------------------------------------


class TestDbInsert:
    def test_returns_inserted_id(self):
        cur = _make_cursor(fetchone=(42,))
        conn = _make_conn(cur)
        with patch("oprim.db_insert.psycopg.connect", return_value=conn):
            result = db_insert(dsn=DSN, table="users", data={"name": "Wiki"})
        assert result == 42

    def test_empty_data_raises_oprim_error(self):
        with pytest.raises(OprimError, match="data must not be empty"):
            db_insert(dsn=DSN, table="users", data={})

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.OperationalError("connection refused")
        conn = _make_conn(cur)
        with patch("oprim.db_insert.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_insert failed"):
                db_insert(dsn=DSN, table="users", data={"name": "Wiki"})

    def test_commit_is_called(self):
        cur = _make_cursor(fetchone=(1,))
        conn = _make_conn(cur)
        with patch("oprim.db_insert.psycopg.connect", return_value=conn):
            db_insert(dsn=DSN, table="users", data={"name": "Wiki"})
        conn.commit.assert_called_once()

    def test_custom_returning_column(self):
        cur = _make_cursor(fetchone=("uuid-abc",))
        conn = _make_conn(cur)
        with patch("oprim.db_insert.psycopg.connect", return_value=conn):
            result = db_insert(dsn=DSN, table="users", data={"name": "Wiki"}, returning="uuid")
        assert result == "uuid-abc"

    def test_returns_none_when_no_row(self):
        cur = _make_cursor(fetchone=None)
        conn = _make_conn(cur)
        with patch("oprim.db_insert.psycopg.connect", return_value=conn):
            result = db_insert(dsn=DSN, table="users", data={"name": "Wiki"})
        assert result is None


# ---------------------------------------------------------------------------
# db_query
# ---------------------------------------------------------------------------


class TestDbQuery:
    def test_returns_list_of_dicts(self):
        rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        cur = _make_cursor(fetchmany=rows)
        conn = _make_conn(cur)
        with patch("oprim.db_query.psycopg.connect", return_value=conn):
            result = db_query(dsn=DSN, query="SELECT * FROM users")
        assert result == rows

    def test_empty_result_returns_empty_list(self):
        cur = _make_cursor(fetchmany=[])
        conn = _make_conn(cur)
        with patch("oprim.db_query.psycopg.connect", return_value=conn):
            result = db_query(dsn=DSN, query="SELECT * FROM users WHERE 1=0")
        assert result == []

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.OperationalError("timeout")
        conn = _make_conn(cur)
        with patch("oprim.db_query.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_query failed"):
                db_query(dsn=DSN, query="SELECT 1")

    def test_params_are_passed_to_execute(self):
        cur = _make_cursor(fetchmany=[{"id": 5}])
        conn = _make_conn(cur)
        params = {"user_id": 5}
        with patch("oprim.db_query.psycopg.connect", return_value=conn):
            db_query(dsn=DSN, query="SELECT * FROM users WHERE id = %(user_id)s", params=params)
        cur.execute.assert_called_once_with("SELECT * FROM users WHERE id = %(user_id)s", params)

    def test_limit_passed_to_fetchmany(self):
        cur = _make_cursor(fetchmany=[])
        conn = _make_conn(cur)
        with patch("oprim.db_query.psycopg.connect", return_value=conn):
            db_query(dsn=DSN, query="SELECT 1", limit=25)
        cur.fetchmany.assert_called_once_with(25)


# ---------------------------------------------------------------------------
# db_write
# ---------------------------------------------------------------------------


class TestDbWrite:
    def test_plain_insert_returns_write_result(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_write.psycopg.connect", return_value=conn):
            result = db_write(dsn=DSN, table="events", data={"name": "click"})
        assert isinstance(result, WriteResult)
        assert result.rows_affected == 1

    def test_upsert_with_conflict_on(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_write.psycopg.connect", return_value=conn):
            result = db_write(
                dsn=DSN,
                table="events",
                data={"slug": "click", "count": 1},
                conflict_on=["slug"],
            )
        assert result.rows_affected == 1
        # verify ON CONFLICT clause appears in the executed query
        call_args = cur.execute.call_args[0]
        assert "ON CONFLICT" in call_args[0]
        assert "DO UPDATE SET" in call_args[0]

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.IntegrityError("duplicate key")
        conn = _make_conn(cur)
        with patch("oprim.db_write.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_write failed"):
                db_write(dsn=DSN, table="events", data={"name": "click"})

    def test_rowcount_reflected_in_result(self):
        cur = _make_cursor(rowcount=0)
        conn = _make_conn(cur)
        with patch("oprim.db_write.psycopg.connect", return_value=conn):
            result = db_write(dsn=DSN, table="events", data={"name": "x"})
        assert result.rows_affected == 0

    def test_empty_data_raises_oprim_error(self):
        with pytest.raises(OprimError, match="data must not be empty"):
            db_write(dsn=DSN, table="events", data={})


# ---------------------------------------------------------------------------
# db_read
# ---------------------------------------------------------------------------


class TestDbRead:
    def test_returns_dict_when_found(self):
        row = {"id": "1", "name": "Alice"}
        cur = _make_cursor(fetchone=row)
        conn = _make_conn(cur)
        with patch("oprim.db_read.psycopg.connect", return_value=conn):
            result = db_read(dsn=DSN, table="users", id="1")
        assert result == row

    def test_returns_none_when_not_found(self):
        cur = _make_cursor(fetchone=None)
        conn = _make_conn(cur)
        with patch("oprim.db_read.psycopg.connect", return_value=conn):
            result = db_read(dsn=DSN, table="users", id="999")
        assert result is None

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.OperationalError("timeout")
        conn = _make_conn(cur)
        with patch("oprim.db_read.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_read failed"):
                db_read(dsn=DSN, table="users", id="1")

    def test_fallback_when_no_deleted_at_column(self):
        """When deleted_at column doesn't exist, falls back to plain query."""
        cur = _make_cursor(fetchone={"id": "1"})
        # first execute raises UndefinedColumn, second succeeds
        cur.execute.side_effect = [
            psycopg.errors.UndefinedColumn("column deleted_at does not exist"),
            None,
        ]
        conn = _make_conn(cur)
        with patch("oprim.db_read.psycopg.connect", return_value=conn):
            result = db_read(dsn=DSN, table="simple", id="1")
        assert result == {"id": "1"}
        assert cur.execute.call_count == 2

    def test_custom_id_column(self):
        row = {"uuid": "abc", "val": 1}
        cur = _make_cursor(fetchone=row)
        conn = _make_conn(cur)
        with patch("oprim.db_read.psycopg.connect", return_value=conn):
            result = db_read(dsn=DSN, table="things", id="abc", id_column="uuid")
        assert result == row
        call_args = cur.execute.call_args[0]
        assert '"uuid"' in call_args[0]


# ---------------------------------------------------------------------------
# db_soft_delete
# ---------------------------------------------------------------------------


class TestDbSoftDelete:
    def test_returns_true_when_row_updated(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_soft_delete.psycopg.connect", return_value=conn):
            result = db_soft_delete(dsn=DSN, table="users", id="1")
        assert result is True

    def test_returns_false_when_row_not_found(self):
        cur = _make_cursor(rowcount=0)
        conn = _make_conn(cur)
        with patch("oprim.db_soft_delete.psycopg.connect", return_value=conn):
            result = db_soft_delete(dsn=DSN, table="users", id="999")
        assert result is False

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.OperationalError("timeout")
        conn = _make_conn(cur)
        with patch("oprim.db_soft_delete.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_soft_delete failed"):
                db_soft_delete(dsn=DSN, table="users", id="1")

    def test_custom_deleted_at_column(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_soft_delete.psycopg.connect", return_value=conn):
            result = db_soft_delete(dsn=DSN, table="items", id="5", deleted_at_column="removed_at")
        assert result is True
        call_args = cur.execute.call_args[0]
        assert '"removed_at"' in call_args[0]


# ---------------------------------------------------------------------------
# db_update
# ---------------------------------------------------------------------------


class TestDbUpdate:
    def test_returns_true_when_row_updated(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_update.psycopg.connect", return_value=conn):
            result = db_update(dsn=DSN, table="users", id="1", data={"name": "Bob"})
        assert result is True

    def test_returns_false_when_row_not_found(self):
        cur = _make_cursor(rowcount=0)
        conn = _make_conn(cur)
        with patch("oprim.db_update.psycopg.connect", return_value=conn):
            result = db_update(dsn=DSN, table="users", id="999", data={"name": "Bob"})
        assert result is False

    def test_db_error_raises_oprim_error(self):
        cur = _make_cursor()
        cur.execute.side_effect = psycopg.OperationalError("lost connection")
        conn = _make_conn(cur)
        with patch("oprim.db_update.psycopg.connect", return_value=conn):
            with pytest.raises(OprimError, match="db_update failed"):
                db_update(dsn=DSN, table="users", id="1", data={"name": "Bob"})

    def test_empty_data_raises_oprim_error(self):
        with pytest.raises(OprimError, match="data must not be empty"):
            db_update(dsn=DSN, table="users", id="1", data={})

    def test_custom_id_column(self):
        cur = _make_cursor(rowcount=1)
        conn = _make_conn(cur)
        with patch("oprim.db_update.psycopg.connect", return_value=conn):
            result = db_update(dsn=DSN, table="items", id="abc", data={"val": 9}, id_column="uuid")
        assert result is True
        call_args = cur.execute.call_args[0]
        assert '"uuid"' in call_args[0]


# ---------------------------------------------------------------------------
# migration_runner
# ---------------------------------------------------------------------------


class TestMigrationRunner:
    def _patch_alembic(self, current_rev="abc123"):
        """Return a dict of patches for alembic internals."""
        mock_engine = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_mig_ctx = MagicMock()
        mock_mig_ctx.get_current_revision.return_value = current_rev
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn_ctx)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn_ctx
        return mock_engine, mock_mig_ctx

    def test_upgrade_returns_migration_result(self):
        mock_engine, mock_mig_ctx = self._patch_alembic("rev1")
        with (
            patch("oprim.migration_runner.command.upgrade") as mock_upgrade,
            patch("oprim.migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim.migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="upgrade",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
            )
        mock_upgrade.assert_called_once()
        assert isinstance(result, MigrationResult)
        assert result.success is True
        assert result.action == "upgrade"
        assert result.current_revision == "rev1"
        assert result.message == "OK"

    def test_downgrade_calls_alembic_downgrade(self):
        mock_engine, mock_mig_ctx = self._patch_alembic(None)
        with (
            patch("oprim.migration_runner.command.downgrade") as mock_down,
            patch("oprim.migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim.migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="downgrade",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
                target="base",
            )
        mock_down.assert_called_once()
        assert result.success is True
        assert result.current_revision is None

    def test_oprim_error_on_alembic_failure(self):
        with patch("oprim.migration_runner.command.upgrade", side_effect=Exception("bad config")):
            with pytest.raises(OprimError, match="migration_runner upgrade failed"):
                migration_runner(
                    action="upgrade",
                    dsn=DSN,
                    migrations_path=Path("/fake/migrations"),
                )

    def test_stamp_action(self):
        mock_engine, mock_mig_ctx = self._patch_alembic("head")
        with (
            patch("oprim.migration_runner.command.stamp") as mock_stamp,
            patch("oprim.migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim.migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="stamp",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
                target="head",
            )
        mock_stamp.assert_called_once()
        assert result.success is True


# ---------------------------------------------------------------------------
# _db_types models
# ---------------------------------------------------------------------------


class TestDbTypes:
    def test_write_result_model(self):
        wr = WriteResult(rows_affected=3)
        assert wr.rows_affected == 3
        assert wr.returned_id is None

    def test_write_result_with_returned_id(self):
        wr = WriteResult(rows_affected=1, returned_id=99)
        assert wr.returned_id == 99

    def test_access_result_model(self):
        ar = AccessResult(user_id="u1", access_tier="gold", action="read", success=True)
        assert ar.user_id == "u1"
        assert ar.success is True
