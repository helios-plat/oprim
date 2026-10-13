"""Tests for Stratum Batch 1 Group 3: migration_runner.

(db_* tests removed: those oprims were deleted in v3.0.0 and moved to obase.persistence.)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from oprim._db_types import AccessResult, WriteResult
from oprim._exceptions import OprimError
from oprim._migration_runner import MigrationResult, migration_runner

DSN = "postgresql://test/testdb"


# ---------------------------------------------------------------------------
# migration_runner
# ---------------------------------------------------------------------------


class TestMigrationRunner:
    def _patch_alembic(self, current_rev="abc123"):
        """Return a dict of patches for alembic internals."""
        mock_engine = MagicMock()
        mock_conn_ctx = MagicMock()
        mock_mig_ctx = MagicMock()
        mock_mig_ctx.get_current_revision.return_value = current_rev
        mock_conn_ctx.__enter__ = MagicMock(return_value=mock_conn_ctx)
        mock_conn_ctx.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn_ctx
        return mock_engine, mock_mig_ctx

    def test_upgrade_returns_migration_result(self):
        mock_engine, mock_mig_ctx = self._patch_alembic("rev1")
        with (
            patch("oprim._migration_runner.command.upgrade") as mock_upgrade,
            patch("oprim._migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim._migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="upgrade",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
            )
        mock_upgrade.assert_called_once()
        assert isinstance(result, MigrationResult)
        assert result.success is True
        assert result.action == "upgrade"
        assert result.current_revision == "rev1"
        assert result.message == "OK"

    def test_downgrade_calls_alembic_downgrade(self):
        mock_engine, mock_mig_ctx = self._patch_alembic(None)
        with (
            patch("oprim._migration_runner.command.downgrade") as mock_down,
            patch("oprim._migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim._migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="downgrade",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
                target="base",
            )
        mock_down.assert_called_once()
        assert result.success is True
        assert result.current_revision is None

    def test_oprim_error_on_alembic_failure(self):
        with patch("oprim._migration_runner.command.upgrade", side_effect=Exception("bad config")):
            with pytest.raises(OprimError, match="migration_runner upgrade failed"):
                migration_runner(
                    action="upgrade",
                    dsn=DSN,
                    migrations_path=Path("/fake/migrations"),
                )

    def test_stamp_action(self):
        mock_engine, mock_mig_ctx = self._patch_alembic("head")
        with (
            patch("oprim._migration_runner.command.stamp") as mock_stamp,
            patch("oprim._migration_runner.create_engine", return_value=mock_engine),
            patch(
                "oprim._migration_runner.MigrationContext.configure",
                return_value=mock_mig_ctx,
            ),
        ):
            result = migration_runner(
                action="stamp",
                dsn=DSN,
                migrations_path=Path("/fake/migrations"),
                target="head",
            )
        mock_stamp.assert_called_once()
        assert result.success is True


# ---------------------------------------------------------------------------
# _db_types models
# ---------------------------------------------------------------------------


class TestDbTypes:
    def test_write_result_model(self):
        wr = WriteResult(rows_affected=3)
        assert wr.rows_affected == 3
        assert wr.returned_id is None

    def test_write_result_with_returned_id(self):
        wr = WriteResult(rows_affected=1, returned_id=99)
        assert wr.returned_id == 99

    def test_access_result_model(self):
        ar = AccessResult(user_id="u1", access_tier="gold", action="read", success=True)
        assert ar.user_id == "u1"
        assert ar.success is True
