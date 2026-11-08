"""The audit trail in Postgres, sharing the connection pool with the LangGraph checkpointer."""

from __future__ import annotations

import logging
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

log = logging.getLogger(__name__)

# One statement per execute: the pool prepares every query (prepare_threshold=0, as LangGraph's
# checkpointer wants), and a prepared statement cannot hold two commands.
_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS audit_events (
        id BIGSERIAL PRIMARY KEY,
        case_id TEXT NOT NULL,
        at TIMESTAMPTZ NOT NULL DEFAULT now(),
        step TEXT NOT NULL,
        event TEXT NOT NULL,
        details JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    "CREATE INDEX IF NOT EXISTS audit_events_case_at ON audit_events (case_id, at)",
)


class PostgresAuditLog:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        async with self._pool.connection() as connection:
            for statement in _SCHEMA:
                await connection.execute(statement)

    async def record(self, case_id: str, step: str, event: str, details: dict[str, Any]) -> None:
        log.info("case=%s step=%s event=%s details=%s", case_id, step, event, details)
        async with self._pool.connection() as connection:
            await connection.execute(
                "INSERT INTO audit_events (case_id, step, event, details) VALUES (%s, %s, %s, %s)",
                (case_id, step, event, Jsonb(details)),
            )

    async def list_for_case(self, case_id: str, limit: int = 200) -> list[dict[str, Any]]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT at, step, event, details FROM audit_events "
                "WHERE case_id = %s ORDER BY at, id LIMIT %s",
                (case_id, limit),
            )
            rows = await cursor.fetchall()
        return [
            {
                "at": row["at"].isoformat(),
                "step": row["step"],
                "event": row["event"],
                "details": row["details"],
            }
            for row in rows
        ]
