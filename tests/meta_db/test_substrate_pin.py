"""Tests for oprim.meta_db.substrate_ops: pin/unpin/list_pinned."""
from __future__ import annotations

from pathlib import Path

import pytest

from oprim.errors import SubstrateNotFoundError
from oprim.meta_db import open_meta_db
from oprim.meta_db.substrate_ops import (
    list_pinned_substrates,
    pin_substrate,
    unpin_substrate,
)

_MIGRATIONS_DIR = (
    Path(__file__).parent.parent.parent / "oprim" / "meta_db" / "migrations"
)

_SUB_FIELDS = "id, ulid, title, mime"


def _setup_db(tmp_path: Path):
    db = open_meta_db(tmp_path / "meta.duckdb")
    db.migrate(_MIGRATIONS_DIR)
    return db


def _insert_substrate(db, sub_id: str, title: str = "Test Doc") -> str:
    db.execute(
        f"INSERT INTO substrate ({_SUB_FIELDS}) VALUES (?, ?, ?, ?)",
        [sub_id, f"ulid_{sub_id}", title, "application/pdf"],
    )
    return sub_id


class TestPinSubstrate:
    def test_pin_sets_is_pinned_true(self, tmp_path: Path):
        db = _setup_db(tmp_path)
        _insert_substrate(db, "sub_01")

        result = pin_substrate(db, "sub_01")

        assert result["is_pinned"] is True
        assert result["substrate_id"] == "sub_01"
        assert result["pinned_at"] is not None
        db.close()

    def test_pin_nonexistent_raises(self, tmp_path: Path):
        db = _setup_db(tmp_path)

        with pytest.raises(SubstrateNotFoundError):
            pin_substrate(db, "nonexistent_id")

        db.close()

    def test_unpin_sets_is_pinned_false(self, tmp_path: Path):
        db = _setup_db(tmp_path)
        _insert_substrate(db, "sub_02")
        pin_substrate(db, "sub_02")

        result = unpin_substrate(db, "sub_02")

        assert result["is_pinned"] is False
        assert result["substrate_id"] == "sub_02"
        db.close()

    def test_unpin_nonexistent_raises(self, tmp_path: Path):
        db = _setup_db(tmp_path)

        with pytest.raises(SubstrateNotFoundError):
            unpin_substrate(db, "nonexistent_id")

        db.close()

    def test_list_pinned_returns_pinned_only(self, tmp_path: Path):
        db = _setup_db(tmp_path)
        _insert_substrate(db, "sub_a", "Doc A")
        _insert_substrate(db, "sub_b", "Doc B")
        _insert_substrate(db, "sub_c", "Doc C")

        pin_substrate(db, "sub_a")
        pin_substrate(db, "sub_c")
        # sub_b stays unpinned

        pinned = list_pinned_substrates(db)

        pinned_ids = {r["id"] for r in pinned}
        assert pinned_ids == {"sub_a", "sub_c"}
        assert "sub_b" not in pinned_ids
        db.close()

    def test_list_pinned_ordered_by_pinned_at_desc(self, tmp_path: Path):
        db = _setup_db(tmp_path)
        for i in range(3):
            _insert_substrate(db, f"sub_{i:02d}", f"Doc {i}")
            pin_substrate(db, f"sub_{i:02d}")

        pinned = list_pinned_substrates(db)

        # Most recently pinned should be first
        assert len(pinned) == 3
        assert pinned[0]["id"] == "sub_02"
        db.close()

    def test_list_pinned_limit(self, tmp_path: Path):
        db = _setup_db(tmp_path)
        for i in range(5):
            _insert_substrate(db, f"sub_{i:02d}")
            pin_substrate(db, f"sub_{i:02d}")

        pinned = list_pinned_substrates(db, limit=3)

        assert len(pinned) == 3
        db.close()

    def test_pin_idempotent(self, tmp_path: Path):
        """Pinning an already-pinned substrate must not raise."""
        db = _setup_db(tmp_path)
        _insert_substrate(db, "sub_idem")
        pin_substrate(db, "sub_idem")

        result = pin_substrate(db, "sub_idem")

        assert result["is_pinned"] is True
        db.close()
