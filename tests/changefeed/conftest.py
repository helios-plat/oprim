"""Fixtures for changefeed tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.meta_db.duckdb import open_meta_db

_MIGRATIONS_DIR = (
    Path(__file__).parent.parent.parent / "oprim" / "meta_db" / "migrations"
)


@pytest.fixture()
def db(tmp_path: Path):
    """Open a fresh MetaDB with all migrations applied."""
    m = open_meta_db(tmp_path / "meta.duckdb")
    m.migrate(_MIGRATIONS_DIR)
    yield m
    m.close()
