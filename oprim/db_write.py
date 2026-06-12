"""oprim-073: db_write — INSERT with optional ON CONFLICT DO UPDATE (upsert)."""

from __future__ import annotations

import warnings
from typing import Any

import psycopg

from oprim._db_types import WriteResult
from oprim._exceptions import OprimError


def db_write(
    *,
    dsn: str,
    table: str,
    data: dict[str, Any],
    conflict_on: list[str] | None = None,
) -> WriteResult:
    """INSERT with optional ON CONFLICT DO UPDATE (upsert).

    Args:
        dsn: PostgreSQL DSN string
        table: Target table name
        data: Column → value mapping
        conflict_on: Columns to use for conflict detection (upsert). None = plain INSERT.

    Returns:
        WriteResult with rows_affected

    Raises:
        OprimError: Database error
    """
    warnings.warn(
        "oprim.db_write is deprecated and will be removed in oprim v3.0.0. "
        "Use obase.persistence.write_one instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not data:
        raise OprimError("db_write: data must not be empty")

    cols = list(data.keys())
    vals = [data[c] for c in cols]
    col_str = ", ".join(f'"{c}"' for c in cols)
    placeholder_str = ", ".join("%s" for _ in cols)

    if conflict_on is None:
        query = f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholder_str})'
    else:
        conflict_cols = ", ".join(f'"{c}"' for c in conflict_on)
        update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in cols)
        query = (
            f'INSERT INTO "{table}" ({col_str}) VALUES ({placeholder_str})'
            f" ON CONFLICT ({conflict_cols}) DO UPDATE SET {update_set}"
        )

    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(query, vals)
            rows_affected = cur.rowcount
            conn.commit()
            return WriteResult(rows_affected=rows_affected)
    except psycopg.Error as e:
        raise OprimError(f"db_write failed: {e}") from e
