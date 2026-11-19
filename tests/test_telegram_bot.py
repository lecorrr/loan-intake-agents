"""Telegram updates through the real dispatcher and service, with the Bot API answered locally."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import UTC, datetime
from typing import Any

import aiohttp
import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.session.base import BaseSession
from aiogram.client.telegram import TelegramAPIServer
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.methods import AnswerCallbackQuery, EditMessageText, GetFile, GetMe, SendMessage
from aiogram.methods.base import TelegramMethod
from aiogram.types import CallbackQuery, Chat, Document, File, Message, PhotoSize, Update, User
from aiohttp import web

from loan_agents.adapters.telegram import case_id_for_chat, chat_id_for_case, decision_callback
from loan_agents.contracts import CaseStatus, Channel, DealStage, DecisionAction
from loan_agents.core import texts_uk as texts
from loan_agents.core.policy import Policy
from loan_agents.fakes import InMemoryCrm, RecordingMessenger
from loan_agents.settings import Settings
from loan_agents.telegram_bot import GET_FILE_TIMEOUT_S, build_dispatcher, polling
from loan_agents.wiring import Runtime, build_runtime
from tests.conftest import TODAY, sample_bytes

CLIENT_CHAT = 42
CLIENT_CASE = case_id_for_chat(CLIENT_CHAT, 1)
MANAGER_CHAT = 7
CLIENT = User(id=CLIENT_CHAT, is_bot=False, first_name="Тарас")
MANAGER = User(id=MANAGER_CHAT, is_bot=False, first_name="Олена")
BOT_USER = User(id=1, is_bot=True, first_name="Loan bot", username="loan_test_bot")
ANSWERS = (
    "Мене звати Тарас Шевченко",
    "067 123 45 67",
    "200 тис грн на 12 місяців",
    "Шкода Октавія 2018 року",
)
# The same client again, for an older car: the case reaches the manager flagged.
SECOND_ANSWERS = (
    "Мене звати Тарас Шевченко",
    "067 123 45 67",
    "100 тис грн на 12 місяців",
    "Рено Логан 2004 року",
)
_ids = itertools.count(1)


class FakeBotApi(BaseSession):
    """Answers Bot API calls without a network and remembers every call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[TelegramMethod[Any]] = []
        self.files: dict[str, bytes] = {}
        self.fail_downloads = False
        self.refuse_callback_answers = False
        self.get_file_errors: list[Exception] = []
        self.timeouts: dict[type, int | None] = {}

    # Both signatures are aiogram's BaseSession interface, timeout parameter included.
    async def make_request(
        self,
        bot: Bot,
        method: TelegramMethod[Any],
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> Any:
        self.calls.append(method)
        self.timeouts[type(method)] = timeout
        if isinstance(method, GetMe):
            return BOT_USER
        if isinstance(method, GetFile) and self.get_file_errors:
            raise self.get_file_errors.pop(0)
        if isinstance(method, GetFile):
            return File(
                file_id=method.file_id, file_unique_id=method.file_id, file_path=method.file_id
            )
        if isinstance(method, AnswerCallbackQuery) and self.refuse_callback_answers:
            raise TelegramBadRequest(method=method, message="query is too old")
        return True

    async def stream_content(
        self,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: int = 30,  # noqa: ASYNC109
        chunk_size: int = 65536,
        raise_for_status: bool = True,
    ) -> AsyncGenerator[bytes, None]:
        if self.fail_downloads:
            raise aiohttp.ClientConnectionError("connection reset")
        yield self.files[url.rsplit("/", 1)[-1]]

    async def close(self) -> None:
        return None

    def calls_of(self, method_type: type) -> list[Any]:
        return [call for call in self.calls if isinstance(call, method_type)]


class Harness:
    def __init__(
        self, runtime: Runtime, api: FakeBotApi, messenger: RecordingMessenger, crm: InMemoryCrm
    ) -> None:
        self.service = runtime.service
        self.api = api
        self.messenger = messenger
        self.crm = crm
        self.bot = Bot("42:TEST", session=api)
        self.dispatcher = build_dispatcher(
            runtime.service,
            runtime.chat_cases,
            self.bot,
            MANAGER_CHAT,
            runtime.settings.max_upload_bytes,
        )

    async def feed(self, update: Update) -> None:
        await self.dispatcher.feed_update(self.bot, update)

    def client_replies(self) -> list[str]:
        return [
            text
            for case_id, text in self.messenger.client_messages
            if chat_id_for_case(case_id) == CLIENT_CHAT
        ]

    def bot_replies(self) -> list[str]:
        return [call.text for call in self.api.calls_of(SendMessage)]


@pytest.fixture
async def harness(settings: Settings) -> AsyncIterator[Harness]:
    messenger = RecordingMessenger()
    crm = InMemoryCrm()
    async with build_runtime(
        settings, fake=True, messenger=messenger, crm=crm, today=lambda: TODAY
    ) as runtime:
        yield Harness(runtime, FakeBotApi(), messenger, crm)


def _message(chat_id: int = CLIENT_CHAT, chat_type: str = "private", **fields: Any) -> Update:
    message = Message(
        message_id=next(_ids),
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=chat_type),
        from_user=CLIENT,
        **fields,
    )
    return Update(update_id=next(_ids), message=message)


def _photo(file_id: str, file_size: int) -> Update:
    size = PhotoSize(
        file_id=file_id, file_unique_id=file_id, width=1200, height=800, file_size=file_size
    )
    return _message(photo=[size])


def _button_press(card_chat_id: int, data: str) -> Update:
    card = Message(
        message_id=next(_ids),
        date=datetime.now(UTC),
        chat=Chat(id=card_chat_id, type="private"),
        from_user=BOT_USER,
        text=f"Нова заявка {CLIENT_CASE}",
    )
    query = CallbackQuery(
        id=str(next(_ids)), from_user=MANAGER, chat_instance="chat", message=card, data=data
    )
    return Update(update_id=next(_ids), callback_query=query)


async def _submit_case(harness: Harness) -> None:
    await harness.service.start(CLIENT_CASE, Channel.TELEGRAM)
    for text in ANSWERS:
        await harness.service.handle_text(CLIENT_CASE, Channel.TELEGRAM, text)
    await harness.service.handle_document(
        CLIENT_CASE, Channel.TELEGRAM, sample_bytes("certificate-ok.png"), "image/png"
    )
    assert (await harness.service.view(CLIENT_CASE)).status is CaseStatus.AWAITING_MANAGER


async def _apply(harness: Harness, command: str, answers: tuple[str, ...], photo: str) -> str:
    """One application through the bot, from the command to the photo; returns its case id."""
    await harness.feed(_message(text=command))
    for text in answers:
        await harness.feed(_message(text=text))
    data = sample_bytes(photo)
    harness.api.files[photo] = data
    await harness.feed(_photo(photo, len(data)))
    return harness.messenger.approval_requests[-1].case_id


async def test_start_in_a_private_chat_opens_a_case(harness: Harness) -> None:
    await harness.feed(_message(text="/start"))
    assert harness.client_replies()[0] == texts.GREETING


async def test_a_chats_first_message_opens_its_case_one(harness: Harness) -> None:
    await harness.feed(_message(text="Добрий день, хочу кредит під заставу авто"))
    assert {case_id for case_id, _ in harness.messenger.client_messages} == {CLIENT_CASE}


async def test_start_carries_on_with_the_open_application(harness: Harness) -> None:
    await harness.feed(_message(text="/start"))
    await harness.feed(_message(text=ANSWERS[0]))
    await harness.feed(_message(text="/start"))
    assert {case_id for case_id, _ in harness.messenger.client_messages} == {CLIENT_CASE}
    assert (await harness.service.view(CLIENT_CASE)).request["full_name"] == "Т. Ш."


async def test_group_chats_are_not_served(harness: Harness) -> None:
    await harness.feed(_message(chat_id=-100123, chat_type="supergroup", text="/start"))
    await harness.feed(_message(chat_id=-100123, chat_type="supergroup", text="Хочу кредит"))
    assert harness.messenger.client_messages == []
    assert harness.api.calls_of(SendMessage) == []


async def test_a_photo_is_downloaded_and_read(harness: Harness) -> None:
    await harness.feed(_message(text="/start"))
    for text in ANSWERS:
        await harness.feed(_message(text=text))
    data = sample_bytes("certificate-ok.png")
    harness.api.files["cert"] = data
    await harness.feed(_photo("cert", len(data)))
    assert harness.client_replies()[-2:] == [texts.DOCUMENT_ACCEPTED, texts.SUBMITTED]
    assert len(harness.messenger.approval_requests) == 1


async def test_an_image_sent_as_a_file_is_read(harness: Harness) -> None:
    await harness.feed(_message(text="/start"))
    data = sample_bytes("certificate-ok.png")
    harness.api.files["file"] = data
    document = Document(
        file_id="file", file_unique_id="file", mime_type="image/png", file_size=len(data)
    )
    await harness.feed(_message(document=document))
    assert texts.DOCUMENT_ACCEPTED in harness.client_replies()


async def test_a_non_image_file_is_refused_without_downloading(harness: Harness) -> None:
    document = Document(file_id="pdf", file_unique_id="pdf", mime_type="application/pdf")
    await harness.feed(_message(document=document))
    assert harness.bot_replies() == [texts.UPLOAD_NOT_IMAGE]
    assert harness.api.calls_of(GetFile) == []


async def test_an_oversized_photo_is_refused_without_downloading(harness: Harness) -> None:
    await harness.feed(_photo("big", 21 * 1024 * 1024))
    assert harness.bot_replies() == [texts.UPLOAD_TOO_LARGE.format(max_mb=10)]
    assert harness.api.calls_of(GetFile) == []


async def test_a_failed_download_asks_for_the_photo_again(harness: Harness) -> None:
    harness.api.fail_downloads = True
    await harness.feed(_photo("cert", 1000))
    assert harness.bot_replies() == [texts.DOWNLOAD_FAILED]


async def test_stickers_and_the_like_get_a_hint(harness: Harness) -> None:
    await harness.feed(_message(contact={"phone_number": "+380671234567", "first_name": "Т"}))
    assert harness.bot_replies() == [texts.UNSUPPORTED_MESSAGE]


async def test_the_manager_chat_decides(harness: Harness) -> None:
    await _submit_case(harness)
    await harness.feed(
        _button_press(MANAGER_CHAT, decision_callback(CLIENT_CASE, DecisionAction.APPROVE))
    )
    assert (await harness.service.view(CLIENT_CASE)).status is CaseStatus.APPROVED
    assert harness.client_replies()[-1] == texts.DECISION_APPROVED
    edits = harness.api.calls_of(EditMessageText)
    assert len(edits) == 1 and edits[0].reply_markup is None


async def test_each_application_from_a_chat_has_a_deal_of_its_own(harness: Harness) -> None:
    first = await _apply(harness, "/start", ANSWERS, "certificate-ok.png")
    await harness.feed(
        _button_press(MANAGER_CHAT, decision_callback(first, DecisionAction.APPROVE))
    )
    second = await _apply(harness, "/new", SECOND_ANSWERS, "certificate-old-car.png")

    assert (await harness.service.view(first)).status is CaseStatus.APPROVED
    assert "OCTAVIA" in harness.crm.deals[first].deal.title
    assert harness.crm.deals[first].stage is DealStage.APPROVED
    assert "LOGAN" in harness.crm.deals[second].deal.title


async def test_a_card_decides_only_the_application_it_shows(harness: Harness) -> None:
    first = await _apply(harness, "/start", ANSWERS, "certificate-ok.png")
    # The client starts over while the first application waits for the manager.
    second = await _apply(harness, "/new", SECOND_ANSWERS, "certificate-old-car.png")
    await harness.feed(_button_press(MANAGER_CHAT, decision_callback(first, DecisionAction.REJECT)))
    assert (await harness.service.view(first)).status is CaseStatus.REJECTED
    assert (await harness.service.view(second)).status is CaseStatus.AWAITING_MANAGER


async def test_buttons_outside_the_manager_chat_are_refused(harness: Harness) -> None:
    await _submit_case(harness)
    await harness.feed(
        _button_press(CLIENT_CHAT, decision_callback(CLIENT_CASE, DecisionAction.APPROVE))
    )
    assert (await harness.service.view(CLIENT_CASE)).status is CaseStatus.AWAITING_MANAGER
    answer = harness.api.calls_of(AnswerCallbackQuery)[0]
    assert answer.text == texts.MANAGER_ONLY
    assert answer.show_alert is True


async def test_a_late_button_press_still_counts(harness: Harness) -> None:
    await _submit_case(harness)
    harness.api.refuse_callback_answers = True
    await harness.feed(
        _button_press(MANAGER_CHAT, decision_callback(CLIENT_CASE, DecisionAction.REJECT))
    )
    assert (await harness.service.view(CLIENT_CASE)).status is CaseStatus.REJECTED


async def test_a_second_press_reports_the_case_as_decided(harness: Harness) -> None:
    await _submit_case(harness)
    press = decision_callback(CLIENT_CASE, DecisionAction.APPROVE)
    await harness.feed(_button_press(MANAGER_CHAT, press))
    await harness.feed(_button_press(MANAGER_CHAT, press))
    assert harness.bot_replies() == [texts.manager_already_decided(CLIENT_CASE)]


@pytest.mark.parametrize("round_number", range(3))
async def test_a_burst_from_one_chat_is_handled_in_order(
    harness: Harness, policy: Policy, round_number: int
) -> None:
    await harness.feed(_message(text="/start"))
    burst = [_message(text=text) for text in ANSWERS[:3]]
    # Polling hands every update to its own task, in update order, all at once.
    await asyncio.gather(*(asyncio.create_task(harness.feed(update)) for update in burst))
    expected = [
        texts.ask_field(field, policy, TODAY.year) for field in ("phone", "amount_uah", "car_make")
    ]
    assert harness.client_replies()[-3:] == expected


async def test_a_network_failure_on_download_gets_one_more_attempt(harness: Harness) -> None:
    await harness.feed(_message(text="/start"))
    for text in ANSWERS:
        await harness.feed(_message(text=text))
    data = sample_bytes("certificate-ok.png")
    harness.api.files["cert"] = data
    harness.api.get_file_errors = [
        TelegramNetworkError(method=GetFile(file_id="cert"), message="timeout")
    ]
    await harness.feed(_photo("cert", len(data)))
    assert texts.DOCUMENT_ACCEPTED in harness.client_replies()
    calls = harness.api.calls_of(GetFile)
    assert len(calls) == 2
    assert harness.api.timeouts[GetFile] == GET_FILE_TIMEOUT_S


@pytest.mark.parametrize(
    ("errors", "attempts"),
    [
        ([TelegramNetworkError(method=GetFile(file_id="x"), message="timeout")] * 2, 2),
        ([TelegramBadRequest(method=GetFile(file_id="x"), message="file is too big")], 1),
    ],
    ids=["network-twice", "refused"],
)
async def test_downloads_give_up_and_ask_again(
    harness: Harness, errors: list[Exception], attempts: int
) -> None:
    harness.api.get_file_errors = list(errors)
    await harness.feed(_photo("cert", 1000))
    assert harness.bot_replies() == [texts.DOWNLOAD_FAILED]
    assert len(harness.api.calls_of(GetFile)) == attempts


class FakeTelegramServer:
    """Telegram's getMe and a long-polling getUpdates over real HTTP, for the polling lifecycle."""

    def __init__(self) -> None:
        self.update_requested = asyncio.Event()
        self.release = asyncio.Event()

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/bot{token}/{method}", self.handle)
        return app

    async def handle(self, request: web.Request) -> web.Response:
        method = request.match_info["method"]
        if method == "getMe":
            user = {"id": 1, "is_bot": True, "first_name": "Loan bot", "username": "loan_test_bot"}
            return web.json_response({"ok": True, "result": user})
        if method == "getUpdates":
            self.update_requested.set()
            with contextlib.suppress(TimeoutError):  # a long poll with nothing new
                await asyncio.wait_for(self.release.wait(), timeout=5)
            return web.json_response({"ok": True, "result": []})
        return web.json_response({"ok": False, "error_code": 404, "description": "Not Found"})


async def test_stopping_the_bot_logs_no_error_and_leaves_no_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = FakeTelegramServer()
    runner = web.AppRunner(server.build_app())
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    session = AiohttpSession(api=TelegramAPIServer.from_base(f"http://127.0.0.1:{port}"))
    bot = Bot("42:TEST", session=session)
    ended: list[asyncio.Task[None]] = []
    try:
        with caplog.at_level(logging.INFO, logger="aiogram"):
            async with polling(Dispatcher(), bot, ended.append):
                await asyncio.wait_for(server.update_requested.wait(), timeout=5)
            await asyncio.sleep(0.2)  # time for a stray task to log, if one were left
        still_polling = [
            task for task in asyncio.all_tasks() if "_polling" in repr(task.get_coro())
        ]
        assert still_polling == []
        assert [record for record in caplog.records if record.levelno >= logging.ERROR] == []
        assert len(ended) == 1 and not ended[0].cancelled() and ended[0].exception() is None
    finally:
        await bot.session.close()
        server.release.set()
        await runner.cleanup()
