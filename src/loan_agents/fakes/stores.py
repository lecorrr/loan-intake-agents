from __future__ import annotations

import uuid
from collections import deque
from typing import Any

_MAX_AUDIT_EVENTS = 10_000


class InMemoryDocumentStore:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], tuple[bytes, str]] = {}

    async def save(self, case_id: str, data: bytes, mime_type: str) -> str:
        doc_id = uuid.uuid4().hex
        self.files[(case_id, doc_id)] = (data, mime_type)
        return doc_id

    async def load(self, case_id: str, doc_id: str) -> tuple[bytes, str]:
        try:
            return self.files[(case_id, doc_id)]
        except KeyError as error:
            raise FileNotFoundError(f"case {case_id}: no document {doc_id}") from error

    async def save_text(self, case_id: str, name: str, text: str) -> str:
        return await self.save(case_id, text.encode("utf-8"), "text/markdown")


class InMemoryAuditLog:
    """Keeps the newest events only, so a long run holds flat memory."""

    def __init__(self) -> None:
        self.events: deque[dict[str, Any]] = deque(maxlen=_MAX_AUDIT_EVENTS)

    async def record(self, case_id: str, step: str, event: str, details: dict[str, Any]) -> None:
        self.events.append({"case_id": case_id, "step": step, "event": event, "details": details})

    def for_case(self, case_id: str) -> list[tuple[str, str]]:
        return [(e["step"], e["event"]) for e in self.events if e["case_id"] == case_id]
