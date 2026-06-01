"""oprim-074: db_read — read a single row by ID with soft-delete filtering."""

from __future__ import annotations

from typing import Any

import psycopg
import psycopg.errors
import psycopg.rows

from oprim._exceptions import OprimError


def db_read(
    *,
    dsn: str,
    table: str,
    id: str,  # noqa: A002
    id_column: str = "id",
) -> dict[str, Any] | None:
    """Read a single row by ID. Returns None if not found or soft-deleted.

    Filters out rows where deleted_at IS NOT NULL if the column exists.

    Args:
        dsn: PostgreSQL DSN
        table: Table name
        id: Row identifier value
        id_column: Column name for the ID (default "id")

    Returns:
        Row as dict, or None if not found
    """
    try:
        with (
            psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn,
            conn.cursor() as cur,
        ):
            sql = f'SELECT * FROM "{table}" WHERE "{id_column}" = %s AND deleted_at IS NULL'
            try:
                cur.execute(sql, (id,))
                return cur.fetchone()
            except psycopg.errors.UndefinedColumn:
                conn.rollback()
                fallback = f'SELECT * FROM "{table}" WHERE "{id_column}" = %s'
                cur.execute(fallback, (id,))
                return cur.fetchone()
    except psycopg.Error as e:
        raise OprimError(f"db_read failed: {e}") from e
