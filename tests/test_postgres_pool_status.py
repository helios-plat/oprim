"""Tests for oprim.postgres_pool_status."""
from __future__ import annotations

from unittest import mock

import pytest

from oprim._exceptions import OprimError
from oprim.postgres_pool_status import postgres_pool_status


def _make_async_conn(rows, max_conn=100, fetch_error=None):
    """Create a mock asyncpg connection."""
    conn = mock.AsyncMock()
    if fetch_error:
        conn.fetch.side_effect = fetch_error
    else:
        conn.fetch.return_value = rows
    conn.fetchval.return_value = str(max_conn) if max_conn else None
    conn.close = mock.AsyncMock()
    return conn


class TestPostgresPoolStatus:
    def test_happy_path_with_apps(self, mock_asyncpg):
        """Normal pool with multiple application_names."""
        rows = [
            {"state": "active", "application_name": "helixa-prob", "count": 8},
            {"state": "idle", "application_name": "helixa-prob", "count": 12},
            {"state": "active", "application_name": "helixa-risk", "count": 3},
            {"state": "idle", "application_name": "helixa-risk", "count": 5},
        ]
        conn = _make_async_conn(rows, max_conn=100)
        mock_asyncpg.return_value = conn

        result = postgres_pool_status(
            host="pg.test",
            username="aegis_readonly",
            password="secret",
        )

        assert result.total_connections == 28
        assert result.active_connections == 11
        assert result.idle_connections == 17
        assert result.by_application["helixa-prob"] == 20
        assert result.max_connections == 100

    def test_pool_exhausted_scenario(self, mock_asyncpg):
        """Pool near max_connections."""
        rows = [
            {"state": "active", "application_name": "helixa-prob", "count": 95},
            {"state": "idle", "application_name": "helixa-prob", "count": 3},
            {"state": "active", "application_name": "helixa-risk", "count": 2},
        ]
        conn = _make_async_conn(rows, max_conn=100)
        mock_asyncpg.return_value = conn

        result = postgres_pool_status(
            host="pg.test",
            username="aegis_readonly",
            password="secret",
        )

        assert result.active_connections == 97
        assert result.total_connections == 100

    def test_idle_in_transaction_detected(self, mock_asyncpg):
        """idle in transaction state is counted separately."""
        rows = [
            {"state": "active", "application_name": "app", "count": 5},
            {"state": "idle in transaction", "application_name": "app", "count": 8},
            {"state": "idle", "application_name": "app", "count": 3},
        ]
        conn = _make_async_conn(rows, max_conn=100)
        mock_asyncpg.return_value = conn

        result = postgres_pool_status(host="pg.test", username="u", password="p")

        assert result.idle_in_transaction == 8

    def test_connection_refused(self, mock_asyncpg):
        """asyncpg.connect raises OSError → OprimError."""
        mock_asyncpg.side_effect = OSError("connection refused")

        with pytest.raises(OprimError, match="Failed to connect"):
            postgres_pool_status(host="pg.test", username="u", password="p")

    def test_query_error(self, mock_asyncpg):
        """asyncpg.PostgresError during fetch raises OprimError."""
        import asyncpg

        conn = _make_async_conn([], fetch_error=asyncpg.PostgresError("query timeout"))
        mock_asyncpg.return_value = conn

        with pytest.raises(OprimError, match="query failed"):
            postgres_pool_status(host="pg.test", username="u", password="p")

    def test_target_database_filter(self, mock_asyncpg):
        """target_database parameter is passed as $1."""
        rows = [{"state": "active", "application_name": "x", "count": 1}]
        conn = _make_async_conn(rows, max_conn=100)
        mock_asyncpg.return_value = conn

        postgres_pool_status(
            host="pg.test",
            username="u",
            password="p",
            target_database="helixa_prod",
        )

        call_args = conn.fetch.call_args
        assert call_args[0][1] == "helixa_prod"
