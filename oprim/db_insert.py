"""oprim-071: db_insert — insert a single row into a PostgreSQL table."""

from __future__ import annotations

import warnings
from typing import Any

import psycopg

from oprim._exceptions import OprimError


def db_insert(
    *,
    dsn: str,
    table: str,
    data: dict[str, Any],
    returning: str = "id",
) -> Any:
    """Insert a single row into the specified table.

    Args:
        dsn: PostgreSQL DSN string
        table: Target table name
        data: Column → value mapping for the new row
        returning: Column to return after insert (default "id")

    Returns:
        Value of the `returning` column for the inserted row

    Raises:
        OprimError: Database error (duplicate key, null constraint, etc.)

    Example:
        >>> db_insert(dsn="postgresql://...", table="users", data={"name": "Wiki"})
        1
    """
    warnings.warn(
        "oprim.db_insert is deprecated and will be removed in oprim v3.0.0. "
        "Use obase.persistence.insert_one instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not data:
        raise OprimError("db_insert: data must not be empty")

    cols = list(data.keys())
    vals = [data[c] for c in cols]
    col_str = ", ".join(f'"{c}"' for c in cols)
    placeholder_str = ", ".join("%s" for _ in cols)
    query = f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholder_str}) RETURNING "{returning}"'

    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(query, vals)
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else None
    except psycopg.Error as e:
        raise OprimError(f"db_insert failed: {e}") from e
