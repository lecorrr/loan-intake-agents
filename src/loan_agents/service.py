"""Runs the agent team one case at a time; every entry point goes through this class.

While a manager decision is pending, client messages get a "waiting" reply and the graph is
left alone: new input would restart the run and repeat the CRM and contract steps.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from PIL import Image, UnidentifiedImageError

from loan_agents.contracts import (
    CASE_ID_PATTERN,
    ApprovalRequest,
    CaseStatus,
    CaseView,
    Channel,
    CrmRefs,
    LoanRequest,
    ManagerDecision,
    TurnResult,
    Valuation,
    VehicleCertificate,
)
from loan_agents.core import masking
from loan_agents.core import texts_uk as texts
from loan_agents.core.routing import TurnInput
from loan_agents.ports import AuditLog, DocumentStore, Messenger, Tracer

log = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_TEXT_CHARS = 2000


class InputError(ValueError):
    """The client sent something the agents can't use. The message is safe to show."""


class CaseNotFound(LookupError):
    pass


class DecisionNotPending(RuntimeError):
    pass


class TurnFailed(RuntimeError):
    """A turn crashed. The client was told; the cause is logged with the case id."""


def validate_case_id(case_id: str) -> str:
    if not CASE_ID_PATTERN.match(case_id):
        raise InputError(f"invalid case id {case_id!r}")
    return case_id


def validate_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise InputError("empty message")
    return cleaned[:MAX_TEXT_CHARS]


def validate_image(data: bytes, mime_type: str, max_bytes: int) -> None:
    if mime_type not in ALLOWED_IMAGE_TYPES:
        raise InputError(texts.UPLOAD_NOT_IMAGE)
    if not data:
        raise InputError(texts.UPLOAD_NOT_IMAGE)
    if len(data) > max_bytes:
        raise InputError(texts.UPLOAD_TOO_LARGE.format(max_mb=max_bytes // (1024 * 1024)))
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise InputError(texts.UPLOAD_NOT_IMAGE) from error


class LoanIntakeService:
    def __init__(
        self,
        graph: CompiledStateGraph,
        documents: DocumentStore,
        audit: AuditLog,
        messenger: Messenger,
        max_upload_bytes: int,
        tracer: Tracer | None = None,
    ) -> None:
        self._graph = graph
        self._documents = documents
        self._audit = audit
        self._messenger = messenger
        self._max_upload_bytes = max_upload_bytes
        self._tracer = tracer
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_users: dict[str, int] = {}

    @asynccontextmanager
    async def _case_lock(self, case_id: str) -> AsyncIterator[None]:
        # Locks are dropped when nobody holds or waits for them, so the dict stays small.
        lock = self._locks.setdefault(case_id, asyncio.Lock())
        self._lock_users[case_id] = self._lock_users.get(case_id, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._lock_users[case_id] -= 1
            if self._lock_users[case_id] == 0:
                del self._lock_users[case_id]
                del self._locks[case_id]

    @staticmethod
    def _config(case_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": case_id}}

    async def _has_pending_decision(self, case_id: str) -> bool:
        snapshot = await self._graph.aget_state(self._config(case_id))
        return bool(snapshot.interrupts)

    async def _waiting(self, case_id: str) -> TurnResult:
        await self._messenger.send_to_client(case_id, [texts.WAITING_FOR_MANAGER])
        return TurnResult(
            case_id=case_id,
            status=CaseStatus.AWAITING_MANAGER,
            replies=[texts.WAITING_FOR_MANAGER],
            awaiting_manager=True,
        )

    async def start(self, case_id: str, channel: Channel) -> TurnResult:
        validate_case_id(case_id)
        async with self._case_lock(case_id):
            if await self._has_pending_decision(case_id):
                return await self._waiting(case_id)
            payload = {
                "case_id": case_id,
                "channel": channel.value,
                "turn_input": TurnInput.START.value,
            }
            return await self._run(case_id, await self._with_defaults(case_id, payload), "start")

    async def handle_text(self, case_id: str, channel: Channel, text: str) -> TurnResult:
        validate_case_id(case_id)
        cleaned = validate_text(text)
        async with self._case_lock(case_id):
            if await self._has_pending_decision(case_id):
                return await self._waiting(case_id)
            payload = {
                "case_id": case_id,
                "channel": channel.value,
                "turn_input": TurnInput.TEXT.value,
                "inbox_text": cleaned,
            }
            return await self._run(case_id, await self._with_defaults(case_id, payload), "message")

    async def handle_document(
        self,
        case_id: str,
        channel: Channel,
        data: bytes,
        mime_type: str,
        caption: str | None = None,
    ) -> TurnResult:
        validate_case_id(case_id)
        validate_image(data, mime_type, self._max_upload_bytes)
        async with self._case_lock(case_id):
            if await self._has_pending_decision(case_id):
                return await self._waiting(case_id)
            doc_id = await self._documents.save(case_id, data, mime_type)
            await self._audit.record(case_id, "service", "document_received", {"doc_id": doc_id})
            payload = {
                "case_id": case_id,
                "channel": channel.value,
                "turn_input": TurnInput.DOCUMENT.value,
                "pending_documents": [{"doc_id": doc_id, "mime_type": mime_type}],
                "inbox_text": validate_text(caption) if caption and caption.strip() else None,
            }
            return await self._run(case_id, await self._with_defaults(case_id, payload), "document")

    async def decide(self, case_id: str, decision: ManagerDecision) -> TurnResult:
        validate_case_id(case_id)
        async with self._case_lock(case_id):
            if not await self._has_pending_decision(case_id):
                raise DecisionNotPending(f"case {case_id} has no decision waiting")
            await self._audit.record(
                case_id, "service", "decision_received", {"action": decision.action.value}
            )
            resume = Command(resume=decision.model_dump(mode="json"))
            return await self._run(case_id, resume, "decision")

    async def reset(self, case_id: str) -> None:
        validate_case_id(case_id)
        async with self._case_lock(case_id):
            checkpointer = self._graph.checkpointer
            if checkpointer is None:
                raise RuntimeError("the graph was compiled without a checkpointer")
            await checkpointer.adelete_thread(case_id)
            await self._audit.record(case_id, "service", "reset", {})

    async def view(self, case_id: str) -> CaseView:
        validate_case_id(case_id)
        snapshot = await self._graph.aget_state(self._config(case_id))
        values = snapshot.values
        if not values:
            raise CaseNotFound(case_id)
        request = LoanRequest.model_validate(values.get("request") or {})
        certificate = values.get("certificate")
        valuation = values.get("valuation")
        crm = values.get("crm")
        return CaseView(
            case_id=case_id,
            status=CaseStatus(values.get("status", CaseStatus.COLLECTING)),
            awaiting_manager=bool(snapshot.interrupts),
            request=masking.masked_request(request),
            certificate=(
                masking.masked_certificate(VehicleCertificate.model_validate(certificate))
                if certificate
                else None
            ),
            valuation=Valuation.model_validate(valuation) if valuation else None,
            crm=CrmRefs.model_validate(crm) if crm else None,
            flags=list(values.get("document_flags") or []),
            transcript=list(values.get("transcript") or []),
        )

    async def _with_defaults(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """A message can arrive before /start (or via the API); give the case its initial state."""
        snapshot = await self._graph.aget_state(self._config(case_id))
        if snapshot.values:
            return payload
        return {"status": CaseStatus.COLLECTING.value, "request": {}} | payload

    async def _run(self, case_id: str, payload: dict[str, Any] | Command, turn: str) -> TurnResult:
        config = self._config(case_id)
        before = await self._graph.aget_state(config)
        sent_before = len(before.values.get("transcript") or [])
        run_config = config | (self._tracer.run_config(case_id, turn) if self._tracer else {})
        try:
            await self._graph.ainvoke(payload, run_config)
        except Exception as error:
            log.exception("case=%s step=service event=turn_failed", case_id)
            await self._audit.record(
                case_id, "service", "turn_failed", {"error": type(error).__name__}
            )
            await self._messenger.send_to_client(case_id, [texts.TECHNICAL_PROBLEM])
            raise TurnFailed(f"case {case_id}: {type(error).__name__}: {error}") from error

        snapshot = await self._graph.aget_state(config)
        replies = list((snapshot.values.get("transcript") or [])[sent_before:])
        await self._messenger.send_to_client(case_id, replies)
        awaiting = bool(snapshot.interrupts)
        if awaiting and not isinstance(payload, Command):
            approval = ApprovalRequest.model_validate(snapshot.interrupts[0].value)
            await self._messenger.send_approval_request(approval)
            await self._audit.record(case_id, "service", "approval_requested", {})
        return TurnResult(
            case_id=case_id,
            status=CaseStatus(snapshot.values.get("status", CaseStatus.COLLECTING)),
            replies=replies,
            awaiting_manager=awaiting,
        )
