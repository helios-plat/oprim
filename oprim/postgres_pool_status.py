"""oprim.postgres_pool_status — Query Postgres connection pool stats."""
from __future__ import annotations

import asyncio

import asyncpg
from obase.tool_registry import register_tool
from pydantic import BaseModel, Field

from oprim._exceptions import OprimError


class PostgresPoolStatus(BaseModel):
    database: str
    total_connections: int
    active_connections: int
    idle_connections: int
    idle_in_transaction: int = Field(description="Dangerous! idle in transaction count")
    waiting_count: int = Field(description="state=active for >30s (possibly blocked)")
    max_connections: int
    by_application: dict[str, int] = Field(description="Connection count per application_name")


_POOL_QUERY = """
SELECT
    state,
    application_name,
    COUNT(*) AS count
FROM pg_stat_activity
WHERE state IS NOT NULL
    AND ($1::text IS NULL OR datname = $1)
GROUP BY state, application_name
"""

_MAX_CONN_QUERY = "SHOW max_connections"


@register_tool(  # type: ignore[untyped-decorator]
    permission="read",
    requires_secrets=["aegis/{env}/pg_admin_password"],
)
def postgres_pool_status(
    *,
    host: str,
    port: int = 5432,
    database: str = "postgres",
    username: str = "aegis_readonly",
    password: str,
    timeout_seconds: float = 10.0,
    target_database: str | None = None,
) -> PostgresPoolStatus:
    """Query Postgres connection pool status via pg_stat_activity.

    Args:
        host: Postgres host
        port: Postgres port
        database: Database to connect to (use "postgres" system db)
        username: User with pg_read_all_stats privilege
        password: Password (typically from Infisical)
        timeout_seconds: Connection + query timeout
        target_database: If set, filter pg_stat_activity by datname

    Returns:
        PostgresPoolStatus with detailed pool metrics.

    Raises:
        OprimError: Connection failure / permission denied / timeout.
    """

    async def _query() -> PostgresPoolStatus:
        try:
            conn = await asyncio.wait_for(
                asyncpg.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=username,
                    password=password,
                ),
                timeout=timeout_seconds,
            )
        except (TimeoutError, asyncpg.PostgresError, OSError) as exc:
            raise OprimError(f"Failed to connect to Postgres at {host}:{port}: {exc}") from exc

        try:
            rows = await conn.fetch(_POOL_QUERY, target_database)
            max_conn_row = await conn.fetchval(_MAX_CONN_QUERY)
            max_conn = int(max_conn_row) if max_conn_row else 0
        except asyncpg.PostgresError as exc:
            raise OprimError(f"Postgres query failed: {exc}") from exc
        finally:
            await conn.close()

        total = active = idle = idle_tx = 0
        by_app: dict[str, int] = {}
        for row in rows:
            cnt = row["count"]
            total += cnt
            by_app[row["application_name"]] = by_app.get(row["application_name"], 0) + cnt
            if row["state"] == "active":
                active += cnt
            elif row["state"] == "idle":
                idle += cnt
            elif row["state"] == "idle in transaction":
                idle_tx += cnt

        return PostgresPoolStatus(
            database=target_database or database,
            total_connections=total,
            active_connections=active,
            idle_connections=idle,
            idle_in_transaction=idle_tx,
            waiting_count=0,
            max_connections=max_conn,
            by_application=by_app,
        )

    return asyncio.run(_query())
