"""Sending to Telegram: client replies and the manager's approval card.

Case ids are "tg-<chat id>-<n>": each application from a chat is its own case, numbered from 1.
"""

from __future__ import annotations

import logging
import re

from aiogram import Bot
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BufferedInputFile, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from loan_agents.contracts import ApprovalRequest, DecisionAction
from loan_agents.core import texts_uk as texts

log = logging.getLogger(__name__)

_CASE_PREFIX = "tg-"
# Older cases are "tg-<chat id>" and may still be waiting for a decision, so both forms parse.
_CASE_ID = re.compile(r"tg-(-?\d+)(?:-(\d+))?")
_CALLBACK_PREFIX = "dec"
_MAX_CALLBACK_BYTES = 64  # Telegram's limit for callback_data


def case_id_for_chat(chat_id: int, number: int) -> str:
    if number < 1:
        raise ValueError(f"case numbers start at 1, got {number}")
    return f"{_CASE_PREFIX}{chat_id}-{number}"


def chat_id_for_case(case_id: str) -> int | None:
    """The chat behind a Telegram case, or None for cases from other channels."""
    if not case_id.startswith(_CASE_PREFIX):
        return None
    match = _CASE_ID.fullmatch(case_id)
    if match is None:
        raise ValueError(f"malformed Telegram case id {case_id!r}")
    return int(match.group(1))


def decision_callback(case_id: str, action: DecisionAction) -> str:
    data = f"{_CALLBACK_PREFIX}:{case_id}:{action.value}"
    if len(data.encode()) > _MAX_CALLBACK_BYTES:
        raise ValueError(f"callback data for {case_id} exceeds {_MAX_CALLBACK_BYTES} bytes")
    return data


def parse_decision_callback(data: str) -> tuple[str, DecisionAction] | None:
    prefix, _, rest = data.partition(":")
    case_id, _, action = rest.rpartition(":")
    if prefix != _CALLBACK_PREFIX or not case_id:
        return None
    try:
        return case_id, DecisionAction(action)
    except ValueError:
        return None


def approval_keyboard(case_id: str) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for action in (DecisionAction.APPROVE, DecisionAction.NEED_DOCUMENTS, DecisionAction.REJECT):
        builder.button(
            text=texts.DECISION_BUTTON[action], callback_data=decision_callback(case_id, action)
        )
    builder.adjust(1)
    return builder.as_markup()


class TelegramMessenger:
    def __init__(self, bot: Bot, manager_chat_id: int) -> None:
        self._bot = bot
        self._manager_chat_id = manager_chat_id

    async def send_to_client(self, case_id: str, texts_to_send: list[str]) -> None:
        chat_id = chat_id_for_case(case_id)
        if chat_id is None:
            return
        for text in texts_to_send:
            try:
                await self._bot.send_message(chat_id, text)
            except TelegramAPIError as error:
                # The reply stays in the case transcript; the turn itself already succeeded.
                log.error("case=%s step=telegram event=send_failed error=%s", case_id, error)

    async def send_approval_request(self, request: ApprovalRequest) -> None:
        await self._bot.send_message(
            self._manager_chat_id, request.card, reply_markup=approval_keyboard(request.case_id)
        )
        if request.contract_text:
            await self._bot.send_document(
                self._manager_chat_id,
                BufferedInputFile(
                    request.contract_text.encode("utf-8"),
                    filename=f"contract-draft-{request.case_id}.md",
                ),
                caption=f"Чернетка договору, заявка {request.case_id}",
            )
