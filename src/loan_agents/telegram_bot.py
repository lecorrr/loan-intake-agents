"""Receiving Telegram updates: client messages and the manager's decision buttons.

Replies go out through the service's messenger, so handlers only route updates into the service.
Clients are served in private chats only; the manager chat may be a group.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable

import aiohttp
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
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


class DownloadFailed(RuntimeError):
    pass


async def _download(bot: Bot, file_id: str) -> bytes:
    try:
        buffer = await bot.download(file_id)
    except (TelegramAPIError, aiohttp.ClientError, TimeoutError) as error:
        raise DownloadFailed(f"file {file_id}: {error}") from error
    if buffer is None:
        raise DownloadFailed(f"file {file_id}: Telegram returned no data")
    return buffer.getvalue()


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
    dispatcher = Dispatcher()
    dispatcher.include_router(_manager_router(service, bot, manager_chat_id))
    dispatcher.include_router(_client_router(service, bot, max_upload_bytes))
    return dispatcher
