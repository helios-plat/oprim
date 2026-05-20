"""Substrate pin/unpin operations on the local DuckDB MetaDB."""
from __future__ import annotations

from datetime import datetime, timezone

from oprim._logging import log as olog
from oprim.errors import SubstrateNotFoundError
from oprim.meta_db.duckdb import MetaDB


def pin_substrate(db: MetaDB, substrate_id: str) -> dict:
    """Pin a substrate so it ranks first in hybrid_search results.

    Returns {"substrate_id", "is_pinned": True, "pinned_at": datetime}.
    Raises SubstrateNotFoundError if the substrate does not exist.
    """
    now = datetime.now(timezone.utc)
    rows = db.fetchall(
        "UPDATE substrate SET is_pinned = TRUE, pinned_at = ?, updated_at = ?"
        " WHERE id = ? RETURNING id, is_pinned, pinned_at",
        [now.isoformat(), now.isoformat(), substrate_id],
    )
    if not rows:
        raise SubstrateNotFoundError(f"substrate {substrate_id!r} not found")

    olog.info("meta_db.substrate_pinned", substrate_id=substrate_id)
    # TODO Phase 2: write changefeed event substrate.pinned
    return {
        "substrate_id": rows[0][0],
        "is_pinned": rows[0][1],
        "pinned_at": rows[0][2],
    }


def unpin_substrate(db: MetaDB, substrate_id: str) -> dict:
    """Unpin a substrate.

    Returns {"substrate_id", "is_pinned": False}.
    Raises SubstrateNotFoundError if the substrate does not exist.
    """
    now = datetime.now(timezone.utc)
    rows = db.fetchall(
        "UPDATE substrate SET is_pinned = FALSE, pinned_at = NULL, updated_at = ?"
        " WHERE id = ? RETURNING id, is_pinned",
        [now.isoformat(), substrate_id],
    )
    if not rows:
        raise SubstrateNotFoundError(f"substrate {substrate_id!r} not found")

    olog.info("meta_db.substrate_unpinned", substrate_id=substrate_id)
    # TODO Phase 2: write changefeed event substrate.unpinned
    return {
        "substrate_id": rows[0][0],
        "is_pinned": rows[0][1],
    }


def list_pinned_substrates(db: MetaDB, limit: int = 50) -> list[dict]:
    """Return all pinned substrates ordered by pinned_at DESC."""
    rows = db.fetchall(
        "SELECT id, title, mime, pinned_at"
        " FROM substrate WHERE is_pinned = TRUE"
        " ORDER BY pinned_at DESC LIMIT ?",
        [limit],
    )
    return [
        {"id": r[0], "title": r[1], "mime": r[2], "pinned_at": r[3]}
        for r in rows
    ]
