"""B2 PostgreSQL alias tests — postgres_long_running_queries / postgres_locks."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from oprim import (
    postgres_long_running_queries,
    postgres_locks,
    postgres_slow_queries,
    postgres_locks_status,
)


def test_postgres_long_running_queries_is_alias():
    assert postgres_long_running_queries is postgres_slow_queries


def test_postgres_locks_is_alias():
    assert postgres_locks is postgres_locks_status


def test_postgres_long_running_queries_callable():
    """Alias is callable and passes kwargs to underlying function."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchall.return_value = []

    with patch("oprim._postgres.psycopg") as mock_psycopg:
        mock_psycopg.connect.return_value = mock_conn
        mock_psycopg.OperationalError = Exception
        result = postgres_long_running_queries(
            dsn="postgresql://user:pass@localhost/db",
            threshold_ms=5000,
        )
    assert isinstance(result, list)


def test_postgres_locks_callable():
    """postgres_locks alias callable."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.cursor.return_value.__enter__ = lambda s: mock_cur
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_cur.fetchall.return_value = []

    with patch("oprim._postgres.psycopg") as mock_psycopg:
        mock_psycopg.connect.return_value = mock_conn
        mock_psycopg.OperationalError = Exception
        result = postgres_locks(dsn="postgresql://user:pass@localhost/db")
    assert isinstance(result, list)


def test_postgres_long_running_queries_has_same_signature():
    """Both functions expose same parameters."""
    import inspect

    sig_orig = inspect.signature(postgres_slow_queries)
    sig_alias = inspect.signature(postgres_long_running_queries)
    assert sig_orig.parameters.keys() == sig_alias.parameters.keys()


def test_postgres_locks_has_same_signature():
    import inspect

    sig_orig = inspect.signature(postgres_locks_status)
    sig_alias = inspect.signature(postgres_locks)
    assert sig_orig.parameters.keys() == sig_alias.parameters.keys()
