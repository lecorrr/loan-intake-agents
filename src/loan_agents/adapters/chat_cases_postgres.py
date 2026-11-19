"""Which case each Telegram chat is on, in Postgres, sharing the pool with the checkpointer."""

from __future__ import annotations

from psycopg_pool import AsyncConnectionPool

_SCHEMA = """
    CREATE TABLE IF NOT EXISTS telegram_chat_cases (
        chat_id BIGINT PRIMARY KEY,
        case_number INTEGER NOT NULL CHECK (case_number > 0),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
"""


class PostgresChatCases:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def setup(self) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(_SCHEMA)

    async def current(self, chat_id: int) -> int | None:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT case_number FROM telegram_chat_cases WHERE chat_id = %s", (chat_id,)
            )
            row = await cursor.fetchone()
        return None if row is None else int(row["case_number"])

    async def start_new(self, chat_id: int) -> int:
        # One statement, so two requests for the same chat can never get the same number.
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                "INSERT INTO telegram_chat_cases (chat_id, case_number) VALUES (%s, 1) "
                "ON CONFLICT (chat_id) DO UPDATE "
                "SET case_number = telegram_chat_cases.case_number + 1, updated_at = now() "
                "RETURNING case_number",
                (chat_id,),
            )
            row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(f"chat {chat_id}: Postgres returned no case number")
        return int(row["case_number"])
