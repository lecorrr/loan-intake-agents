"""The one place where adapters are chosen and connected to the service."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date

from aiogram import Bot
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from loan_agents.adapters.audit_postgres import PostgresAuditLog
from loan_agents.adapters.chat_cases_postgres import PostgresChatCases
from loan_agents.adapters.espocrm import EspoClient, EspoCrm, api_key_http
from loan_agents.adapters.files import FileDocumentStore
from loan_agents.adapters.llm_langchain import LangChainExtractor
from loan_agents.adapters.logging_messenger import LoggingMessenger
from loan_agents.adapters.prices_csv import CsvPriceBook, PriceTableError
from loan_agents.adapters.telegram import TelegramMessenger
from loan_agents.adapters.tracing_langfuse import open_langfuse
from loan_agents.agents import AgentDeps, build_graph
from loan_agents.fakes import (
    FakeExtractor,
    InMemoryAuditLog,
    InMemoryChatCases,
    InMemoryCrm,
    InMemoryDocumentStore,
)
from loan_agents.ports import (
    AuditLog,
    ChatCases,
    CrmGateway,
    DocumentStore,
    Extractor,
    Messenger,
    Tracer,
)
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
    chat_cases: ChatCases
    bot: Bot | None


def check_crm_settings(settings: Settings) -> None:
    """A CRM address without a key is a setup left half done; the app refuses to start on it."""
    if settings.espocrm_url is not None and settings.espocrm_api_key is None:
        raise ConfigError(
            "ESPOCRM_URL is set but ESPOCRM_API_KEY is empty; "
            "run `loan-agents crm-setup --write-env`, or empty ESPOCRM_URL to keep deals in memory"
        )


async def _open_crm(
    stack: AsyncExitStack, settings: Settings, today: Callable[[], date]
) -> CrmGateway:
    check_crm_settings(settings)
    if settings.espocrm_url is None or settings.espocrm_api_key is None:
        log.warning("step=wiring event=no_crm hint='set ESPOCRM_URL'; deals stay in memory")
        return InMemoryCrm()
    http = await stack.enter_async_context(
        api_key_http(settings.espocrm_url, settings.espocrm_api_key.get_secret_value())
    )
    return EspoCrm(
        EspoClient(http), settings.espocrm_stages, site_url=settings.espocrm_url, today=today
    )


def _open_tracing(stack: AsyncExitStack, settings: Settings) -> Tracer:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        raise ConfigError("tracing needs both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    tracer = open_langfuse(
        settings.langfuse_base_url,
        settings.langfuse_public_key,
        settings.langfuse_secret_key.get_secret_value(),
    )
    # Sends the traces still queued when the app stops.
    stack.push_async_callback(asyncio.to_thread, tracer.shutdown)
    log.info("step=wiring event=tracing url=%s", settings.langfuse_base_url)
    return tracer


async def _open_postgres(
    stack: AsyncExitStack, conninfo: str
) -> tuple[BaseCheckpointSaver, AuditLog, ChatCases]:
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
    chat_cases = PostgresChatCases(pool)
    await chat_cases.setup()
    return checkpointer, audit, chat_cases


@asynccontextmanager
async def build_runtime(
    settings: Settings,
    *,
    fake: bool = False,
    messenger: Messenger | None = None,
    crm: CrmGateway | None = None,
    extractor: Extractor | None = None,
    tracer: Tracer | None = None,
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
        chat_cases: ChatCases
        checkpointer: BaseCheckpointSaver
        bot: Bot | None = None

        if fake:
            extractor = extractor or FakeExtractor()
            crm = crm or InMemoryCrm()
            documents = InMemoryDocumentStore()
            checkpointer, audit = InMemorySaver(), InMemoryAuditLog()
            chat_cases = InMemoryChatCases()
        else:
            extractor = extractor or LangChainExtractor(
                settings.llm_model,
                ollama_base_url=settings.ollama_base_url,
                timeout_s=settings.llm_timeout_s,
            )
            if crm is None:
                crm = await _open_crm(stack, settings, today)
            documents = FileDocumentStore(settings.data_dir / "uploads")
            if settings.database_url is not None:
                checkpointer, audit, chat_cases = await _open_postgres(
                    stack, settings.database_url.get_secret_value()
                )
            else:
                log.warning("step=wiring event=no_database hint='cases are lost on restart'")
                checkpointer, audit = InMemorySaver(), InMemoryAuditLog()
                chat_cases = InMemoryChatCases()

        if messenger is None:
            if settings.telegram_bot_token is not None and settings.manager_chat_id is not None:
                bot = Bot(token=settings.telegram_bot_token.get_secret_value())
                stack.push_async_callback(bot.session.close)
                messenger = TelegramMessenger(bot, settings.manager_chat_id)
            else:
                messenger = LoggingMessenger()

        if tracer is None and settings.langfuse_secret_key is not None:
            tracer = _open_tracing(stack, settings)

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
        service = LoanIntakeService(
            graph, documents, audit, messenger, settings.max_upload_bytes, tracer
        )
        yield Runtime(
            settings=settings,
            service=service,
            crm=crm,
            extractor=extractor,
            messenger=messenger,
            chat_cases=chat_cases,
            bot=bot,
        )
