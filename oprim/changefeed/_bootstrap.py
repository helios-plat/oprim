"""Lazy schema bootstrap for changefeed tables.

Consumers (e.g. oskill.sync tests, cold-start sync) may open a MetaDB without
running oprim migrations. ``ensure_schema`` idempotently applies the canonical
changefeed DDL so the writer/reader/snapshot work regardless of migration state.

Single source of truth: the DDL lives only in
``oprim/meta_db/migrations/003_phase2_changefeed.sql``; this module reads and
executes it (no duplicated schema definition).
"""

from __future__ import annotations

from pathlib import Path

from oprim.meta_db.duckdb import MetaDB

_DDL_FILE = (
    Path(__file__).resolve().parent.parent / "meta_db" / "migrations" / "003_phase2_changefeed.sql"
)

_ENSURED_ATTR = "_changefeed_schema_ready"


def ensure_schema(db: MetaDB) -> None:
    """Apply the canonical changefeed DDL to *db* once per instance."""
    if getattr(db, _ENSURED_ATTR, False):
        return
    text = _DDL_FILE.read_text(encoding="utf-8")
    # strip SQL line comments so split-on-";" yields runnable statements
    sql = "\n".join(line for line in text.splitlines() if not line.strip().startswith("--"))
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            db.execute(stmt)
    setattr(db, _ENSURED_ATTR, True)
