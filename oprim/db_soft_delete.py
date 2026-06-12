"""oprim-075: db_soft_delete — mark a row as deleted by setting deleted_at = NOW()."""

from __future__ import annotations

import warnings

import psycopg

from oprim._exceptions import OprimError


def db_soft_delete(
    *,
    dsn: str,
    table: str,
    id: str,  # noqa: A002
    deleted_at_column: str = "deleted_at",
) -> bool:
    """Mark a row as deleted by setting deleted_at = NOW(). Does not physically delete.

    Args:
        dsn: PostgreSQL DSN
        table: Table name
        id: Row ID (string, matched against "id" column)
        deleted_at_column: Name of the soft-delete timestamp column

    Returns:
        True if row was found and updated, False if not found
    """
    warnings.warn(
        "oprim.db_soft_delete is deprecated and will be removed in oprim v3.0.0. "
        "Use obase.persistence.soft_delete_one instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    query = (
        f'UPDATE "{table}" SET "{deleted_at_column}" = NOW()'
        f' WHERE "id" = %s AND "{deleted_at_column}" IS NULL'
    )
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(query, (id,))
            rows_affected = cur.rowcount
            conn.commit()
            return rows_affected > 0
    except psycopg.Error as e:
        raise OprimError(f"db_soft_delete failed: {e}") from e
