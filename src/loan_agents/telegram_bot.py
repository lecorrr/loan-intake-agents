"""Receiving Telegram updates: client messages and the manager's decision buttons.

Replies go out through the service's messenger, so handlers only route updates into the service.
Clients are served in private chats only; the manager chat may be a group.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError, TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.types import CallbackQuery, Message

from loan_agents.adapters.telegram import case_id_for_chat, parse_decision_callback
from loan_agents.contracts import Channel, ManagerDecision, TurnResult
from loan_agents.core import texts_uk as texts
from loan_agents.service import (
    ALLOWED_IMAGE_TYPES,
    DecisionNotPending,
    InputError,
    LoanIntakeService,
    TurnFailed,
)

log = logging.getLogger(__name__)

# The Bot API's getFile refuses anything bigger, whatever our own limit is.
TELEGRAM_DOWNLOAD_LIMIT_BYTES = 20 * 1024 * 1024
# A getFile call once hung for aiogram's default 60 s; a fresh attempt usually answers at once.
GET_FILE_TIMEOUT_S = 10
FILE_DOWNLOAD_TIMEOUT_S = 20
DOWNLOAD_ATTEMPTS = 2


class DownloadFailed(RuntimeError):
    pass


async def _download_once(bot: Bot, file_id: str) -> bytes:
    file = await bot.get_file(file_id, request_timeout=GET_FILE_TIMEOUT_S)
    if not file.file_path:
        raise DownloadFailed(f"file {file_id}: Telegram gave no file path")
    buffer = await bot.download_file(file.file_path, timeout=FILE_DOWNLOAD_TIMEOUT_S)
    if buffer is None:
        raise DownloadFailed(f"file {file_id}: Telegram returned no data")
    return buffer.getvalue()


async def _download(bot: Bot, file_id: str) -> bytes:
    """Network failures get one more attempt; Telegram refusing the file does not."""
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            return await _download_once(bot, file_id)
        except (TelegramNetworkError, aiohttp.ClientError, TimeoutError) as error:
            if attempt == DOWNLOAD_ATTEMPTS:
                raise DownloadFailed(f"file {file_id}: {error}") from error
            log.warning("step=telegram event=download_retry attempt=%d error=%s", attempt, error)
        except TelegramAPIError as error:
            raise DownloadFailed(f"file {file_id}: {error}") from error
    raise AssertionError("unreachable: the last attempt returns or raises")


async def _run_turn(message: Message, turn: Awaitable[TurnResult]) -> None:
    try:
        await turn
    except InputError as error:
        await message.answer(str(error))
    except TurnFailed:
        pass  # The service logged the cause and already apologised to the client.


async def _acknowledge(callback: CallbackQuery, text: str, *, alert: bool = False) -> None:
    # A button pressed while the bot was down arrives late, and Telegram then refuses the answer.
    # The press itself still counts, so a failed acknowledgement must not stop the decision.
    try:
        await callback.answer(text, show_alert=alert)
    except TelegramAPIError as error:
        log.warning("step=telegram event=callback_answer_failed error=%s", error)


def _client_router(service: LoanIntakeService, bot: Bot, max_upload_bytes: int) -> Router:
    router = Router(name="clients")
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    limit_bytes = min(max_upload_bytes, TELEGRAM_DOWNLOAD_LIMIT_BYTES)
    too_large = texts.UPLOAD_TOO_LARGE.format(max_mb=limit_bytes // (1024 * 1024))

    async def receive_upload(
        message: Message, file_id: str, file_size: int | None, mime_type: str
    ) -> None:
        if file_size and file_size > limit_bytes:
            await message.answer(too_large)
            return
        case_id = case_id_for_chat(message.chat.id)
        try:
            data = await _download(bot, file_id)
        except DownloadFailed as error:
            log.warning("case=%s step=telegram event=download_failed error=%s", case_id, error)
            await message.answer(texts.DOWNLOAD_FAILED)
            return
        turn = service.handle_document(
            case_id, Channel.TELEGRAM, data, mime_type, caption=message.caption
        )
        await _run_turn(message, turn)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        case_id = case_id_for_chat(message.chat.id)
        await _run_turn(message, service.start(case_id, Channel.TELEGRAM))

    @router.message(Command("new"))
    async def on_new(message: Message) -> None:
        case_id = case_id_for_chat(message.chat.id)
        await service.reset(case_id)
        await _run_turn(message, service.start(case_id, Channel.TELEGRAM))

    @router.message(F.photo)
    async def on_photo(message: Message) -> None:
        photo = message.photo[-1]  # Telegram lists sizes smallest first
        # Telegram re-encodes photos as JPEG.
        await receive_upload(message, photo.file_id, photo.file_size, "image/jpeg")

    @router.message(F.document)
    async def on_document(message: Message) -> None:
        document = message.document
        if document is None or (document.mime_type or "") not in ALLOWED_IMAGE_TYPES:
            await message.answer(texts.UPLOAD_NOT_IMAGE)
            return
        await receive_upload(
            message, document.file_id, document.file_size, document.mime_type or ""
        )

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        case_id = case_id_for_chat(message.chat.id)
        await _run_turn(message, service.handle_text(case_id, Channel.TELEGRAM, message.text or ""))

    @router.message()
    async def on_other(message: Message) -> None:
        await message.answer(texts.UNSUPPORTED_MESSAGE)

    return router


def _manager_router(service: LoanIntakeService, bot: Bot, manager_chat_id: int) -> Router:
    router = Router(name="manager")

    @router.callback_query(F.data.startswith("dec:"))
    async def on_decision(callback: CallbackQuery) -> None:
        parsed = parse_decision_callback(callback.data or "")
        card = callback.message
        if parsed is None or card is None:
            await _acknowledge(callback, texts.UNKNOWN_ACTION)
            return
        if card.chat.id != manager_chat_id:
            await _acknowledge(callback, texts.MANAGER_ONLY, alert=True)
            return
        case_id, action = parsed
        await _acknowledge(callback, texts.SAVING_DECISION)
        decision = ManagerDecision(action=action, manager=f"telegram:{callback.from_user.id}")
        try:
            await service.decide(case_id, decision)
        except DecisionNotPending:
            await bot.send_message(manager_chat_id, texts.manager_already_decided(case_id))
            return
        except TurnFailed:
            await bot.send_message(manager_chat_id, texts.manager_decision_failed(case_id))
            return
        log.info("case=%s step=telegram event=decision action=%s", case_id, action)
        if isinstance(card, Message) and card.text:
            try:
                await card.edit_text(texts.decided_card(card.text, action), reply_markup=None)
            except TelegramAPIError as error:
                # The decision is saved; only the card still shows the buttons.
                log.warning("case=%s step=telegram event=card_edit_failed error=%s", case_id, error)

    return router


def build_dispatcher(
    service: LoanIntakeService,
    bot: Bot,
    manager_chat_id: int,
    max_upload_bytes: int,
) -> Dispatcher:
    # aiogram runs synchronous filters in a thread pool, so updates from one chat can overtake
    # each other on the way to a handler. Event isolation takes a per-chat lock before any filter
    # runs, which keeps each chat's updates in arrival order; different chats still run in
    # parallel. It keeps one small lock per chat for the life of the process.
    dispatcher = Dispatcher(events_isolation=SimpleEventIsolation())
    dispatcher.include_router(_manager_router(service, bot, manager_chat_id))
    dispatcher.include_router(_client_router(service, bot, max_upload_bytes))
    return dispatcher


@asynccontextmanager
async def polling(
    dispatcher: Dispatcher, bot: Bot, on_end: Callable[[asyncio.Task[None]], None]
) -> AsyncIterator[asyncio.Task[None]]:
    """Polls Telegram for the life of the block. The caller owns the bot and closes its session.

    Stopping goes through stop_polling: cancelling the start_polling task instead leaves aiogram's
    inner polling task running against a closed session, which logs a false ERROR on shutdown.
    """
    task = asyncio.create_task(
        dispatcher.start_polling(bot, handle_signals=False, close_bot_session=False),
        name="telegram-polling",
    )
    task.add_done_callback(on_end)
    try:
        yield task
    finally:
        if not task.done():
            try:
                await dispatcher.stop_polling()
            except RuntimeError:  # start_polling has not begun yet, so nothing runs inside it
                task.cancel()
            await asyncio.wait([task])  # its outcome is reported by on_end
