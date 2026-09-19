"""meta_db — pluggable metadata store.

Backend is chosen by ``META_DB_BACKEND`` env (default ``duckdb`` = legacy, so
existing consumers are unchanged). ``postgres`` routes to the shared aii-postgres
(see postgres.PgMetaDB) so oskill's ingest persists to the same store stratum
reads instead of a private DuckDB file.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oprim.meta_db.duckdb import MetaDB

__all__ = ["MetaDB", "open_meta_db"]


def __getattr__(name: str):
    if name == "MetaDB":
        from oprim.meta_db.duckdb import MetaDB

        return MetaDB
    raise AttributeError(name)


def open_meta_db(path: Path):
    """Open a MetaDB. Postgres backend when META_DB_BACKEND=postgres, else DuckDB."""
    backend = os.environ.get("META_DB_BACKEND", "duckdb").strip().lower()
    if backend in ("postgres", "pg", "postgresql"):
        from oprim.meta_db.postgres import PgMetaDB

        return PgMetaDB(path)
    from oprim.meta_db.duckdb import open_meta_db as _open_duckdb

    return _open_duckdb(path)
