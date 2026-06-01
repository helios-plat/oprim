"""oprim-072: db_query — execute a parameterized SELECT and return rows as dicts."""

from __future__ import annotations

from typing import Any

import psycopg
import psycopg.rows

from oprim._exceptions import OprimError


def db_query(
    *,
    dsn: str,
    query: str,
    params: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Execute a parameterized SELECT query and return rows as dicts.

    Args:
        dsn: PostgreSQL DSN string
        query: SQL query string (use %(name)s placeholders)
        params: Named parameters for the query
        limit: Maximum rows to return (default 100)

    Returns:
        List of row dicts

    Raises:
        OprimError: Database error or invalid query
    """
    try:
        with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn, conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchmany(limit)
    except psycopg.Error as e:
        raise OprimError(f"db_query failed: {e}") from e
