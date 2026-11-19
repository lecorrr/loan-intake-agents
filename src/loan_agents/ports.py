"""Interfaces to the outside world. Agents and the service depend on these, never on adapters."""

from __future__ import annotations

from typing import Any, Protocol

from loan_agents.contracts import (
    ApprovalRequest,
    CertificateReading,
    CrmDeal,
    CrmRefs,
    DealStage,
    RequestExtraction,
)


class ExtractorError(RuntimeError):
    """The model could not be reached or returned something unusable."""


class CrmError(RuntimeError):
    """The CRM refused a call or could not be reached."""


class Extractor(Protocol):
    async def extract_request(self, text: str) -> RequestExtraction: ...

    async def read_certificate(self, image: bytes, mime_type: str) -> CertificateReading: ...


class PriceBook(Protocol):
    def find_price_uah(self, make: str, model: str, year: int) -> int | None: ...


class CrmGateway(Protocol):
    async def upsert_deal(self, deal: CrmDeal) -> CrmRefs:
        """Creates the contact and deal, or updates the deal already linked to this case."""
        ...

    async def add_comment(self, deal_id: str, text: str) -> None: ...

    async def set_stage(self, deal_id: str, stage: DealStage) -> None: ...


class DocumentStore(Protocol):
    async def save(self, case_id: str, data: bytes, mime_type: str) -> str:
        """Returns the new document's id."""
        ...

    async def load(self, case_id: str, doc_id: str) -> tuple[bytes, str]:
        """Returns the bytes and the mime type."""
        ...

    async def save_text(self, case_id: str, name: str, text: str) -> str: ...


class AuditLog(Protocol):
    async def record(self, case_id: str, step: str, event: str, details: dict[str, Any]) -> None:
        """Details must already be free of unmasked personal data."""
        ...


class Messenger(Protocol):
    async def send_to_client(self, case_id: str, texts: list[str]) -> None: ...

    async def send_approval_request(self, request: ApprovalRequest) -> None: ...


class Tracer(Protocol):
    def run_config(self, case_id: str, turn: str) -> dict[str, Any]:
        """Extra LangChain run config (callbacks, name, metadata) for one turn of a case."""
        ...


class ChatCases(Protocol):
    """Which case each Telegram chat is on. A chat's cases are numbered from 1."""

    async def current(self, chat_id: int) -> int | None:
        """The number of the chat's current case, or None before its first."""
        ...

    async def start_new(self, chat_id: int) -> int:
        """Moves the chat on to its next case and returns that case's number."""
        ...
