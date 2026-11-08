"""Cases and the audit trail in a real Postgres. Skipped unless a database is given, e.g. the one
docker compose starts:

    LOAN_AGENTS_TEST_DATABASE_URL=postgresql://loan:loan@127.0.0.1:5433/loan \
        pytest tests/test_postgres.py
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
import pytest
from pydantic import SecretStr

from loan_agents.contracts import CaseStatus, Channel, DecisionAction, ManagerDecision
from loan_agents.fakes import FakeExtractor, InMemoryCrm, RecordingMessenger
from loan_agents.service import CaseNotFound
from loan_agents.settings import Settings
from loan_agents.wiring import Runtime, build_runtime
from tests.conftest import TODAY, sample_bytes

DATABASE_URL = os.environ.get("LOAN_AGENTS_TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="set LOAN_AGENTS_TEST_DATABASE_URL to run against Postgres"
)

ANSWERS = (
    "Мене звати Тарас Шевченко",
    "067 123 45 67",
    "200 тис грн на 12 місяців",
    "Шкода Октавія 2018 року",
)


@pytest.fixture
def case_id() -> str:
    return f"web-pg-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def pg_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"database_url": SecretStr(DATABASE_URL)})


@asynccontextmanager
async def running(settings: Settings, crm: InMemoryCrm) -> AsyncIterator[Runtime]:
    # Not fake: the Postgres checkpointer, audit log and file store are the real adapters.
    async with build_runtime(
        settings,
        extractor=FakeExtractor(),
        crm=crm,
        messenger=RecordingMessenger(),
        today=lambda: TODAY,
    ) as runtime:
        yield runtime


async def _audit_events(case_id: str) -> list[str]:
    async with await psycopg.AsyncConnection.connect(DATABASE_URL) as connection:
        cursor = await connection.execute(
            "SELECT event FROM audit_events WHERE case_id = %s ORDER BY id", (case_id,)
        )
        return [row[0] for row in await cursor.fetchall()]


async def _delete_audit(case_id: str) -> None:
    async with await psycopg.AsyncConnection.connect(DATABASE_URL, autocommit=True) as connection:
        await connection.execute("DELETE FROM audit_events WHERE case_id = %s", (case_id,))


async def test_a_case_survives_a_restart(pg_settings: Settings, case_id: str) -> None:
    crm = InMemoryCrm()
    async with running(pg_settings, crm) as runtime:
        await runtime.service.start(case_id, Channel.WEB)
        for text in ANSWERS:
            await runtime.service.handle_text(case_id, Channel.WEB, text)
        submitted = await runtime.service.handle_document(
            case_id, Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )
        assert submitted.awaiting_manager

    try:
        async with running(pg_settings, crm) as runtime:
            assert (await runtime.service.view(case_id)).status is CaseStatus.AWAITING_MANAGER
            decided = await runtime.service.decide(
                case_id, ManagerDecision(action=DecisionAction.APPROVE, manager="test")
            )
            assert decided.status is CaseStatus.APPROVED
        assert len(await _audit_events(case_id)) > 0
    finally:
        async with running(pg_settings, crm) as runtime:
            await runtime.service.reset(case_id)
        await _delete_audit(case_id)


async def test_reset_deletes_the_stored_case(pg_settings: Settings, case_id: str) -> None:
    try:
        async with running(pg_settings, InMemoryCrm()) as runtime:
            await runtime.service.start(case_id, Channel.WEB)
            await runtime.service.reset(case_id)
        async with running(pg_settings, InMemoryCrm()) as runtime:
            with pytest.raises(CaseNotFound):
                await runtime.service.view(case_id)
    finally:
        await _delete_audit(case_id)
