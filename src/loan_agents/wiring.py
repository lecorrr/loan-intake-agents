"""The one place where adapters are chosen and connected to the service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date

import httpx
from aiogram import Bot
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from loan_agents.adapters.audit_postgres import PostgresAuditLog
from loan_agents.adapters.bitrix24 import Bitrix24Client, Bitrix24Crm
from loan_agents.adapters.files import FileDocumentStore
from loan_agents.adapters.llm_langchain import LangChainExtractor
from loan_agents.adapters.logging_messenger import LoggingMessenger
from loan_agents.adapters.prices_csv import CsvPriceBook, PriceTableError
from loan_agents.adapters.telegram import TelegramMessenger
from loan_agents.agents import AgentDeps, build_graph
from loan_agents.contracts import DealStage
from loan_agents.fakes import FakeExtractor, InMemoryAuditLog, InMemoryCrm, InMemoryDocumentStore
from loan_agents.ports import AuditLog, CrmGateway, DocumentStore, Extractor, Messenger
from loan_agents.service import LoanIntakeService
from loan_agents.settings import ConfigError, Settings, load_policy, load_template

log = logging.getLogger(__name__)

POOL_OPEN_TIMEOUT_S = 15


@dataclass
class Runtime:
    settings: Settings
    service: LoanIntakeService
    crm: CrmGateway
    extractor: Extractor
    messenger: Messenger
    bot: Bot | None


def _bitrix24_stages(settings: Settings) -> dict[DealStage, str]:
    return {
        DealStage.NEW: settings.b24_stage_new,
        DealStage.REVIEW: settings.b24_stage_review,
        DealStage.APPROVED: settings.b24_stage_approved,
        DealStage.REJECTED: settings.b24_stage_rejected,
    }


async def _open_postgres(
    stack: AsyncExitStack, conninfo: str
) -> tuple[BaseCheckpointSaver, AuditLog]:
    pool = AsyncConnectionPool(
        conninfo,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open(wait=True, timeout=POOL_OPEN_TIMEOUT_S)
    stack.push_async_callback(pool.close)
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    audit = PostgresAuditLog(pool)
    await audit.setup()
    return checkpointer, audit


@asynccontextmanager
async def build_runtime(
    settings: Settings,
    *,
    fake: bool = False,
    messenger: Messenger | None = None,
    crm: CrmGateway | None = None,
    extractor: Extractor | None = None,
    today: Callable[[], date] = date.today,
) -> AsyncIterator[Runtime]:
    """fake=True runs everything in memory with the rule-based extractor: no model, no network."""
    policy = load_policy(settings.config_dir / "policy.yaml")
    try:
        prices = CsvPriceBook.load(settings.config_dir / "market_prices.csv")
    except PriceTableError as error:
        raise ConfigError(str(error)) from error
    template = load_template(settings.templates_dir / "loan_agreement_uk.md.j2")

    async with AsyncExitStack() as stack:
        documents: DocumentStore
        audit: AuditLog
        checkpointer: BaseCheckpointSaver
        bot: Bot | None = None

        if fake:
            extractor = extractor or FakeExtractor()
            crm = crm or InMemoryCrm()
            documents = InMemoryDocumentStore()
            checkpointer, audit = InMemorySaver(), InMemoryAuditLog()
        else:
            extractor = extractor or LangChainExtractor(
                settings.llm_model,
                ollama_base_url=settings.ollama_base_url,
                timeout_s=settings.llm_timeout_s,
            )
            if crm is None and settings.b24_webhook_url:
                http = await stack.enter_async_context(httpx.AsyncClient(timeout=30))
                crm = Bitrix24Crm(
                    Bitrix24Client(settings.b24_webhook_url, http), _bitrix24_stages(settings)
                )
            elif crm is None:
                log.warning("step=wiring event=no_crm hint='set B24_WEBHOOK_URL'; using memory")
                crm = InMemoryCrm()
            documents = FileDocumentStore(settings.data_dir / "uploads")
            if settings.database_url is not None:
                checkpointer, audit = await _open_postgres(
                    stack, settings.database_url.get_secret_value()
                )
            else:
                log.warning("step=wiring event=no_database hint='cases are lost on restart'")
                checkpointer, audit = InMemorySaver(), InMemoryAuditLog()

        if messenger is None:
            if settings.telegram_bot_token is not None and settings.manager_chat_id is not None:
                bot = Bot(token=settings.telegram_bot_token.get_secret_value())
                stack.push_async_callback(bot.session.close)
                messenger = TelegramMessenger(bot, settings.manager_chat_id)
            else:
                messenger = LoggingMessenger()

        graph = build_graph(
            AgentDeps(
                extractor=extractor,
                prices=prices,
                crm=crm,
                documents=documents,
                audit=audit,
                policy=policy,
                contract_template=template,
                today=today,
            ),
            checkpointer,
        )
        service = LoanIntakeService(graph, documents, audit, messenger, settings.max_upload_bytes)
        yield Runtime(
            settings=settings,
            service=service,
            crm=crm,
            extractor=extractor,
            messenger=messenger,
            bot=bot,
        )
