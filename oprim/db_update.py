"""oprim-076: db_update — update a single row by ID."""

from __future__ import annotations

from typing import Any

import psycopg

from oprim._exceptions import OprimError


def db_update(
    *,
    dsn: str,
    table: str,
    id: str,  # noqa: A002
    data: dict[str, Any],
    id_column: str = "id",
) -> bool:
    """Update a single row by ID.

    Args:
        dsn: PostgreSQL DSN
        table: Table name
        id: Row identifier value
        data: Columns and new values to set
        id_column: ID column name (default "id")

    Returns:
        True if row was found and updated, False if not found
    """
    if not data:
        raise OprimError("db_update: data must not be empty")

    cols = list(data.keys())
    vals = [data[c] for c in cols]
    set_clause = ", ".join(f'"{c}" = %s' for c in cols)
    query = f'UPDATE "{table}" SET {set_clause} WHERE "{id_column}" = %s'
    vals.append(id)

    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(query, vals)
            rows_affected = cur.rowcount
            conn.commit()
            return rows_affected > 0
    except psycopg.Error as e:
        raise OprimError(f"db_update failed: {e}") from e
