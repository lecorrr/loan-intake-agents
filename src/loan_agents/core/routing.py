"""The supervisor's decision: which agent works next, given the case state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from loan_agents.contracts import CaseStatus


class TurnInput(StrEnum):
    START = "start"
    TEXT = "text"
    DOCUMENT = "document"


class Step(StrEnum):
    INTAKE = "intake"
    DOCUMENTS = "documents"
    ASK = "ask"
    ASK_DOCUMENT = "ask_document"
    VALUATION = "valuation"
    CRM = "crm"
    CONTRACT = "contract"
    SUBMIT = "submit"
    CLOSED = "closed"
    WAIT = "wait"


@dataclass(frozen=True)
class RouteFacts:
    status: CaseStatus
    turn_input: TurnInput
    has_unread_text: bool
    has_unread_document: bool
    missing_fields: tuple[str, ...]
    has_certificate: bool
    has_valuation: bool
    has_crm_deal: bool
    needs_contract: bool


def next_step(facts: RouteFacts) -> Step:
    if facts.status in (CaseStatus.APPROVED, CaseStatus.REJECTED):
        return Step.CLOSED
    if facts.status is CaseStatus.AWAITING_MANAGER:
        return Step.WAIT
    # Text first, so a photo's caption is read before the photo is cross-checked.
    if facts.has_unread_text:
        return Step.INTAKE
    if facts.has_unread_document:
        return Step.DOCUMENTS
    if facts.missing_fields:
        return Step.ASK
    if not facts.has_certificate:
        # A rejected photo already got its own explanation this turn.
        if facts.turn_input is TurnInput.DOCUMENT:
            return Step.WAIT
        return Step.ASK_DOCUMENT
    if not facts.has_valuation:
        return Step.VALUATION
    if not facts.has_crm_deal:
        return Step.CRM
    if facts.needs_contract:
        return Step.CONTRACT
    return Step.SUBMIT
