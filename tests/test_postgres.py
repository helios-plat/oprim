"""Tests for oprim PostgreSQL operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from oprim import (
    postgres_locks_status,
    postgres_pool_status,
    postgres_replication_lag,
    postgres_slow_queries,
    postgres_table_size,
)
from oprim._exceptions import OprimAuthError, OprimConnectionError, OprimError
from oprim._postgres import LockInfo, PoolStatus, ReplicationLag, SlowQuery, TableSize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_psycopg_connect(fetchall_returns=None, fetchone_returns=None, side_effect=None):
    """Return a patched psycopg.connect context."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)

    if fetchall_returns is not None:
        cursor.fetchall.side_effect = fetchall_returns if isinstance(fetchall_returns, list) else [fetchall_returns]
    if fetchone_returns is not None:
        cursor.fetchone.side_effect = fetchone_returns if isinstance(fetchone_returns, list) else [fetchone_returns]

    conn.cursor.return_value = cursor
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)

    if side_effect:
        return patch("oprim._postgres.psycopg.connect", side_effect=side_effect)
    return patch("oprim._postgres.psycopg.connect", return_value=conn)


# ---------------------------------------------------------------------------
# postgres_pool_status
# ---------------------------------------------------------------------------

class TestPostgresPoolStatus:
    def test_healthy_pool(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [("active", 3), ("idle", 7)]
        cur.fetchone.side_effect = [("100",), (0,)]
        conn.cursor.return_value = cur

        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_pool_status(dsn="postgresql://u:p@localhost/db")

        assert isinstance(result, PoolStatus)
        assert result.active_connections == 3
        assert result.idle_connections == 7
        assert result.max_connections == 100
        assert result.total_connections == 10
        assert result.usage_percent == pytest.approx(10.0)

    def test_high_usage(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [("active", 90)]
        cur.fetchone.side_effect = [("100",), (0,)]
        conn.cursor.return_value = cur

        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_pool_status(dsn="postgresql://u:p@localhost/db")

        assert result.usage_percent > 80

    def test_idle_in_transaction_present(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = [("active", 2), ("idle in transaction", 3)]
        cur.fetchone.side_effect = [("200",), (0,)]
        conn.cursor.return_value = cur

        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_pool_status(dsn="postgresql://u:p@localhost/db")

        assert result.idle_in_transaction == 3

    def test_connection_refused(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("connection refused")
            with pytest.raises(OprimConnectionError):
                postgres_pool_status(dsn="postgresql://u:p@badhost/db")

    def test_auth_failure(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("password authentication failed")
            with pytest.raises(OprimAuthError):
                postgres_pool_status(dsn="postgresql://u:wrongpass@localhost/db")

    def test_psycopg_not_installed(self):
        with patch("oprim._postgres.psycopg", None):
            with pytest.raises(OprimError, match="psycopg"):
                postgres_pool_status(dsn="postgresql://u:p@localhost/db")

    def test_timeout_error(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = TimeoutError("timed out")
            with pytest.raises(OprimError):
                postgres_pool_status(dsn="postgresql://u:p@localhost/db")

    def test_unexpected_exception(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = RuntimeError("unexpected")
            with pytest.raises(OprimConnectionError):
                postgres_pool_status(dsn="postgresql://u:p@localhost/db")


# ---------------------------------------------------------------------------
# postgres_slow_queries
# ---------------------------------------------------------------------------

class TestPostgresSlowQueries:
    def _conn_with_slow_queries(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        # First call: extension check; second: slow queries
        cur.fetchone.side_effect = [(1,), None]
        cur.fetchall.return_value = rows
        conn.cursor.return_value = cur
        return conn

    def test_with_slow_queries(self):
        rows = [(1001, "SELECT * FROM big_table", 50, 50000.0, 1000.5, 5000.0, 100)]
        conn = self._conn_with_slow_queries(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_slow_queries(dsn="postgresql://u:p@localhost/db", threshold_ms=500)
        assert len(result) == 1
        assert isinstance(result[0], SlowQuery)
        assert result[0].mean_time_ms == pytest.approx(1000.5)

    def test_no_slow_queries(self):
        conn = self._conn_with_slow_queries([])
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_slow_queries(dsn="postgresql://u:p@localhost/db")
        assert result == []

    def test_threshold_boundary(self):
        rows = [(1, "SELECT 1", 1, 1000.0, 1000.0, 1000.0, 1)]
        conn = self._conn_with_slow_queries(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_slow_queries(dsn="postgresql://u:p@localhost/db", threshold_ms=1000)
        assert len(result) == 1

    def test_pg_stat_statements_not_installed(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchone.return_value = None  # extension not found
        conn.cursor.return_value = cur

        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            with pytest.raises(OprimError, match="pg_stat_statements"):
                postgres_slow_queries(dsn="postgresql://u:p@localhost/db")

    def test_connection_error(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("refused")
            with pytest.raises(OprimConnectionError):
                postgres_slow_queries(dsn="postgresql://u:p@badhost/db")


# ---------------------------------------------------------------------------
# postgres_locks_status
# ---------------------------------------------------------------------------

class TestPostgresLocksStatus:
    def _make_lock_conn(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = rows
        conn.cursor.return_value = cur
        return conn

    def test_waiting_locks(self):
        rows = [("relation", "my_table", "RowExclusiveLock", False, 1234, "SELECT ...", "Lock", 5.0)]
        conn = self._make_lock_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_locks_status(dsn="postgresql://u:p@localhost/db")
        assert len(result) == 1
        assert isinstance(result[0], LockInfo)
        assert result[0].granted is False
        assert result[0].mode == "RowExclusiveLock"

    def test_no_locks(self):
        conn = self._make_lock_conn([])
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_locks_status(dsn="postgresql://u:p@localhost/db")
        assert result == []

    def test_include_granted(self):
        rows = [("relation", None, "AccessShareLock", True, 100, None, None, None)]
        conn = self._make_lock_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_locks_status(dsn="postgresql://u:p@localhost/db", include_granted=True)
        assert result[0].granted is True

    def test_multiple_locks(self):
        rows = [
            ("relation", "tbl1", "RowExclusiveLock", False, 1, "Q1", "Lock", 10.0),
            ("relation", "tbl2", "ShareLock", False, 2, "Q2", "Lock", 2.0),
        ]
        conn = self._make_lock_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_locks_status(dsn="postgresql://u:p@localhost/db")
        assert len(result) == 2

    def test_connection_error(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("refused")
            with pytest.raises(OprimConnectionError):
                postgres_locks_status(dsn="postgresql://u:p@badhost/db")


# ---------------------------------------------------------------------------
# postgres_table_size
# ---------------------------------------------------------------------------

class TestPostgresTableSize:
    def _make_conn(self, rows):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchall.return_value = rows
        conn.cursor.return_value = cur
        return conn

    def test_normal_tables(self):
        rows = [("public", "users", 1_000_000, 800_000, 150_000, 50_000, 5000)]
        conn = self._make_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_table_size(dsn="postgresql://u:p@localhost/db")
        assert len(result) == 1
        assert isinstance(result[0], TableSize)
        assert result[0].table_name == "users"
        assert result[0].total_bytes == 1_000_000

    def test_empty_schema(self):
        conn = self._make_conn([])
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_table_size(dsn="postgresql://u:p@localhost/db")
        assert result == []

    def test_custom_schema(self):
        rows = [("analytics", "events", 500_000, 400_000, 80_000, 20_000, 2000)]
        conn = self._make_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_table_size(dsn="postgresql://u:p@localhost/db", schema="analytics")
        assert result[0].schema_name == "analytics"

    def test_multiple_tables_returned(self):
        rows = [
            ("public", "orders", 2_000_000, 1_500_000, 400_000, 100_000, 10000),
            ("public", "users", 500_000, 400_000, 80_000, 20_000, 2000),
        ]
        conn = self._make_conn(rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_table_size(dsn="postgresql://u:p@localhost/db")
        assert len(result) == 2

    def test_connection_error(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("refused")
            with pytest.raises(OprimConnectionError):
                postgres_table_size(dsn="postgresql://u:p@badhost/db")


# ---------------------------------------------------------------------------
# postgres_replication_lag
# ---------------------------------------------------------------------------

class TestPostgresReplicationLag:
    def _make_conn(self, is_replica: bool, replicas):
        conn = MagicMock()
        cur = MagicMock()
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        cur.fetchone.return_value = (is_replica,)
        cur.fetchall.return_value = replicas
        conn.cursor.return_value = cur
        return conn

    def test_single_node(self):
        conn = self._make_conn(is_replica=False, replicas=[])
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_replication_lag(dsn="postgresql://u:p@localhost/db")
        assert isinstance(result, ReplicationLag)
        assert result.is_primary is True
        assert result.replicas == []
        assert result.max_lag_seconds is None

    def test_with_replicas(self):
        replica_rows = [("10.0.0.2", "streaming", "async", 0.5, 1024)]
        conn = self._make_conn(is_replica=False, replicas=replica_rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_replication_lag(dsn="postgresql://u:p@primary/db")
        assert result.is_primary is True
        assert len(result.replicas) == 1
        assert result.max_lag_seconds == pytest.approx(0.5)

    def test_lag_spike(self):
        replica_rows = [("10.0.0.2", "streaming", "async", 120.0, 50_000_000)]
        conn = self._make_conn(is_replica=False, replicas=replica_rows)
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_replication_lag(dsn="postgresql://u:p@primary/db")
        assert result.max_lag_seconds == pytest.approx(120.0)

    def test_on_replica(self):
        conn = self._make_conn(is_replica=True, replicas=[])
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.connect.return_value = conn
            mock_psycopg.OperationalError = Exception
            result = postgres_replication_lag(dsn="postgresql://u:p@replica/db")
        assert result.is_primary is False

    def test_connection_refused(self):
        with patch("oprim._postgres.psycopg") as mock_psycopg:
            mock_psycopg.OperationalError = type("OpError", (Exception,), {})
            mock_psycopg.connect.side_effect = mock_psycopg.OperationalError("refused")
            with pytest.raises(OprimConnectionError):
                postgres_replication_lag(dsn="postgresql://u:p@badhost/db")
