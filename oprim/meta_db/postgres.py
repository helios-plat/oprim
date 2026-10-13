"""PostgreSQL backend for meta_db — mirrors the DuckDB MetaDB surface.

Selected via ``META_DB_BACKEND=postgres`` (default stays DuckDB, so existing
consumers are unaffected). Ported from stratum.db's proven ``_ConnWrapper``:
translates DuckDB-style ``?`` / ``$name`` placeholders, connects to the shared
aii-postgres via ``STRATUM_PG_*`` env with ``search_path=stratum`` — so oskill's
ingest writes land in the SAME Postgres store that stratum reads, instead of a
private DuckDB file.

Background: DuckDB was retired from this deployment. oskill.ingest_substrate /
detect_duplicate went on calling ``open_meta_db`` (DuckDB), whose file became
table-less after the migration → substrate persistence silently died. Making
``open_meta_db`` return this PG backend restores durable persistence and removes
the DuckDB/PG split-brain entirely.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from oprim.errors import MetaDBError

try:  # psycopg2 is only needed when this backend is actually selected
    import psycopg2
    import psycopg2.extras
    import psycopg2.pool
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]

_DOLLAR_RE = re.compile(r"\$(\w+)")

_pool: Any = None
_pool_lock = threading.Lock()


def _dsn_kwargs() -> dict[str, Any]:
    return {
        "host": os.environ.get("STRATUM_PG_HOST", "127.0.0.1"),
        "port": int(os.environ.get("STRATUM_PG_PORT", "5435")),
        "user": os.environ.get("STRATUM_PG_USER", "aii"),
        "password": os.environ.get("STRATUM_PG_PASSWORD", ""),
        "dbname": os.environ.get("STRATUM_PG_DB", "aii_kg"),
        # Unqualified table names resolve to the stratum schema (matches stratum.db).
        "options": "-c search_path=" + os.environ.get("STRATUM_PG_SCHEMA", "stratum"),
    }


def _get_pool() -> Any:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if psycopg2 is None:
                    raise MetaDBError(
                        "psycopg2 required for META_DB_BACKEND=postgres. "
                        "Install with: pip install 'psycopg2-binary'"
                    )
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    int(os.environ.get("STRATUM_PG_POOL_MIN", "1")),
                    int(os.environ.get("STRATUM_PG_POOL_MAX", "10")),
                    **_dsn_kwargs(),
                )
    return _pool


def _translate(sql: str, params: Any) -> str:
    """DuckDB placeholders → psycopg2. `?`→`%s` (positional), `$name`→`%(name)s`."""
    if params is not None and isinstance(params, (list, tuple)) and "?" in sql:
        return sql.replace("?", "%s")
    if "$" in sql:
        return _DOLLAR_RE.sub(r"%(\1)s", sql)
    return sql


class PgMetaDB:
    """PG-backed, MetaDB-compatible wrapper: execute / fetchall / close / migrate.

    Autocommit-per-statement matches DuckDB's semantics the callers were written
    against. jsonb columns are returned as RAW STRINGS (matching DuckDB) so the
    many callers that do ``json.loads(col)`` keep working.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._pool = _get_pool()
        self._conn = self._pool.getconn()
        self._conn.autocommit = True
        psycopg2.extras.register_default_jsonb(conn_or_curs=self._conn, loads=lambda x: x)

    def connect(self) -> Any:
        return self._conn

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        cur = self._conn.cursor()
        try:
            cur.execute(_translate(sql, params), params)
        except Exception as e:  # mirror DuckDB MetaDB.execute error surface
            raise MetaDBError(f"Execute failed: {e}") from e
        return cur

    def fetchall(self, sql: str, params: list[Any] | None = None) -> list[tuple]:
        return self.execute(sql, params).fetchall()

    def migrate(self, migrations_dir: Path) -> None:
        # Postgres schema is owned/migrated by stratum (db/pg_migrations); no-op here.
        return

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._pool.putconn(self._conn)
            finally:
                self._conn = None
