"""Tests for oprim.meta_db.postgres (PgMetaDB).

Two tiers:
  * Pure-logic (no DB): placeholder translation, DSN/env parsing, error surface,
    interface contract, backend routing. These MUST pass everywhere.
  * Integration (real PG): auto-skips unless STRATUM_PG_DSN is set — same opt-in
    shape as obase's test_crud. Exercises a real execute/fetchall round trip and
    the jsonb-as-RAW-STRING contract.

The placeholder-translation tests are the load-bearing ones: a bug there does not
fail loudly (insert error) — it silently rewrites production SQL and writes the
wrong data. So they pin the exact `?`/`$name` rewrite behaviour, including the
guards that leave SQL untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import oprim.meta_db as meta_db
import oprim.meta_db.postgres as pg
from oprim.errors import MetaDBError
from oprim.meta_db.postgres import PgMetaDB, _dsn_kwargs, _translate

_STRATUM_ENV = (
    "STRATUM_PG_HOST",
    "STRATUM_PG_PORT",
    "STRATUM_PG_USER",
    "STRATUM_PG_PASSWORD",
    "STRATUM_PG_DB",
    "STRATUM_PG_SCHEMA",
)


# ---------------------------------------------------------------------------
# Placeholder translation — the safety-critical path
# ---------------------------------------------------------------------------
class TestTranslate:
    def test_positional_question_marks_become_percent_s(self):
        assert (
            _translate("INSERT INTO t VALUES (?, ?)", [1, "a"]) == "INSERT INTO t VALUES (%s, %s)"
        )

    def test_positional_with_tuple_params(self):
        assert _translate("SELECT * FROM t WHERE x = ?", (7,)) == ("SELECT * FROM t WHERE x = %s")

    def test_every_question_mark_is_rewritten(self):
        sql = "INSERT INTO substrate (id, ulid, title, mime) VALUES (?, ?, ?, ?)"
        out = _translate(sql, ["a", "b", "c", "d"])
        assert out == ("INSERT INTO substrate (id, ulid, title, mime) VALUES (%s, %s, %s, %s)")
        assert "?" not in out

    def test_named_dollar_becomes_pyformat(self):
        assert _translate("SELECT * FROM t WHERE name = $name", {"name": "x"}) == (
            "SELECT * FROM t WHERE name = %(name)s"
        )

    def test_multiple_distinct_named_params(self):
        out = _translate("... WHERE a = $a AND b = $b", {"a": 1, "b": 2})
        assert out == "... WHERE a = %(a)s AND b = %(b)s"

    def test_named_param_with_underscores_and_digits(self):
        # \w+ must consume the whole identifier, not stop at the first char.
        assert _translate("WHERE k = $user_id2", {"user_id2": 5}) == ("WHERE k = %(user_id2)s")

    def test_no_placeholder_sql_is_untouched(self):
        sql = "SELECT count(*) FROM substrate"
        assert _translate(sql, None) == sql

    def test_question_mark_without_seq_params_is_left_as_is(self):
        # Guard: `?` is only rewritten when list/tuple params are supplied. A `?`
        # with no bind params has nothing to bind, so it is passed through verbatim
        # rather than being corrupted into %s.
        assert _translate("SELECT ?", None) == "SELECT ?"

    def test_dict_params_do_not_trigger_positional_rewrite(self):
        # dict params are for $name style; a stray `?` must not become %s here.
        assert _translate("SELECT ?", {"x": 1}) == "SELECT ?"


# ---------------------------------------------------------------------------
# DSN / env parsing
# ---------------------------------------------------------------------------
class TestDsnKwargs:
    def test_defaults_when_env_unset(self, monkeypatch):
        for var in _STRATUM_ENV:
            monkeypatch.delenv(var, raising=False)
        kw = _dsn_kwargs()
        assert kw["host"] == "127.0.0.1"
        assert kw["port"] == 5435
        assert kw["user"] == "aii"
        assert kw["password"] == ""
        assert kw["dbname"] == "aii_kg"
        assert kw["options"] == "-c search_path=stratum"

    def test_port_is_coerced_to_int(self, monkeypatch):
        monkeypatch.setenv("STRATUM_PG_PORT", "6543")
        kw = _dsn_kwargs()
        assert kw["port"] == 6543
        assert isinstance(kw["port"], int)

    def test_env_overrides_are_applied(self, monkeypatch):
        monkeypatch.setenv("STRATUM_PG_HOST", "db.internal")
        monkeypatch.setenv("STRATUM_PG_USER", "svc")
        monkeypatch.setenv("STRATUM_PG_PASSWORD", "s3cret")
        monkeypatch.setenv("STRATUM_PG_DB", "kg_prod")
        kw = _dsn_kwargs()
        assert kw["host"] == "db.internal"
        assert kw["user"] == "svc"
        assert kw["password"] == "s3cret"
        assert kw["dbname"] == "kg_prod"

    def test_schema_flows_into_search_path_option(self, monkeypatch):
        monkeypatch.setenv("STRATUM_PG_SCHEMA", "helivex")
        assert _dsn_kwargs()["options"] == "-c search_path=helivex"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_get_pool_raises_when_psycopg2_missing(self, monkeypatch):
        monkeypatch.setattr(pg, "psycopg2", None)
        monkeypatch.setattr(pg, "_pool", None)
        with pytest.raises(MetaDBError, match="psycopg2"):
            pg._get_pool()

    def test_execute_wraps_driver_errors_in_metadberror(self):
        db = PgMetaDB.__new__(PgMetaDB)  # bypass __init__ (no real pool/connection)
        cur = MagicMock()
        cur.execute.side_effect = RuntimeError("boom")
        conn = MagicMock()
        conn.cursor.return_value = cur
        db._conn = conn
        with pytest.raises(MetaDBError, match="Execute failed"):
            db.execute("SELECT 1")

    def test_fetchall_also_surfaces_metadberror(self):
        db = PgMetaDB.__new__(PgMetaDB)
        cur = MagicMock()
        cur.execute.side_effect = RuntimeError("boom")
        conn = MagicMock()
        conn.cursor.return_value = cur
        db._conn = conn
        with pytest.raises(MetaDBError):
            db.fetchall("SELECT 1")


# ---------------------------------------------------------------------------
# Interface contract + backend routing
# ---------------------------------------------------------------------------
class TestInterfaceContract:
    @pytest.mark.parametrize("method", ["execute", "fetchall", "connect", "close", "migrate"])
    def test_pgmetadb_exposes_metadb_surface(self, method):
        assert callable(getattr(PgMetaDB, method))

    def test_migrate_is_a_noop(self):
        db = PgMetaDB.__new__(PgMetaDB)  # migrate must not touch the connection
        assert db.migrate(Path("/does/not/exist")) is None

    def test_close_without_connection_is_safe(self):
        db = PgMetaDB.__new__(PgMetaDB)
        db._conn = None
        db.close()  # must not raise

    def test_close_returns_conn_to_pool_and_is_idempotent(self):
        db = PgMetaDB.__new__(PgMetaDB)
        pool = MagicMock()
        conn = MagicMock()
        db._pool = pool
        db._conn = conn
        db.close()
        pool.putconn.assert_called_once_with(conn)
        assert db._conn is None
        db.close()  # second close is a no-op
        pool.putconn.assert_called_once()

    @pytest.mark.parametrize("backend", ["postgres", "pg", "postgresql", "POSTGRES"])
    def test_open_meta_db_routes_to_postgres(self, monkeypatch, backend):
        monkeypatch.setenv("META_DB_BACKEND", backend)
        sentinel = object()
        monkeypatch.setattr(pg, "PgMetaDB", lambda path: sentinel)
        assert meta_db.open_meta_db(Path("ignored")) is sentinel

    def test_open_meta_db_defaults_to_duckdb(self, monkeypatch):
        monkeypatch.delenv("META_DB_BACKEND", raising=False)
        sentinel = object()
        monkeypatch.setattr(meta_db, "_open_duckdb", lambda path: sentinel)
        assert meta_db.open_meta_db(Path("meta.duckdb")) is sentinel

    def test_open_meta_db_unknown_backend_falls_back_to_duckdb(self, monkeypatch):
        monkeypatch.setenv("META_DB_BACKEND", "sqlite")
        sentinel = object()
        monkeypatch.setattr(meta_db, "_open_duckdb", lambda path: sentinel)
        assert meta_db.open_meta_db(Path("meta.duckdb")) is sentinel


# ---------------------------------------------------------------------------
# Integration — real PostgreSQL (opt-in via STRATUM_PG_DSN)
# ---------------------------------------------------------------------------
_DSN = os.environ.get("STRATUM_PG_DSN")


@pytest.mark.skipif(not _DSN, reason="STRATUM_PG_DSN not set; skipping real-PG integration")
class TestPgIntegration:
    @pytest.fixture
    def db(self, monkeypatch):
        from urllib.parse import urlparse

        u = urlparse(_DSN)
        monkeypatch.setenv("STRATUM_PG_HOST", u.hostname or "127.0.0.1")
        monkeypatch.setenv("STRATUM_PG_PORT", str(u.port or 5432))
        monkeypatch.setenv("STRATUM_PG_USER", u.username or "")
        monkeypatch.setenv("STRATUM_PG_PASSWORD", u.password or "")
        monkeypatch.setenv("STRATUM_PG_DB", (u.path or "/").lstrip("/"))
        monkeypatch.setattr(pg, "_pool", None)  # rebuild pool against this DSN
        db = PgMetaDB()
        yield db
        db.close()

    def test_execute_and_fetchall_roundtrip(self, db):
        db.execute("CREATE TEMP TABLE t_pg_meta (x int, y text)")
        db.execute("INSERT INTO t_pg_meta VALUES (?, ?)", [7, "hi"])
        rows = db.fetchall("SELECT x, y FROM t_pg_meta WHERE x = ?", [7])
        assert rows == [(7, "hi")]

    def test_jsonb_is_returned_as_raw_string(self, db):
        db.execute("CREATE TEMP TABLE t_pg_json (id int, meta jsonb)")
        db.execute("INSERT INTO t_pg_json VALUES (?, ?)", [1, '{"a": 1}'])
        rows = db.fetchall("SELECT meta FROM t_pg_json WHERE id = ?", [1])
        val = rows[0][0]
        assert isinstance(val, str)  # RAW STRING (DuckDB parity), not a dict
        assert json.loads(val) == {"a": 1}
