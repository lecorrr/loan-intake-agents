"""Messenger used when no Telegram bot is set: messages go to the log only."""

from __future__ import annotations

import logging

from loan_agents.contracts import ApprovalRequest

log = logging.getLogger(__name__)


class LoggingMessenger:
    async def send_to_client(self, case_id: str, texts: list[str]) -> None:
        for text in texts:
            log.info("case=%s step=messenger event=client_reply chars=%d", case_id, len(text))

    async def send_approval_request(self, request: ApprovalRequest) -> None:
        log.warning(
            "case=%s step=messenger event=approval_waiting "
            "hint='no Telegram bot configured; decide with POST /api/cases/%s/decision'",
            request.case_id,
            request.case_id,
        )
