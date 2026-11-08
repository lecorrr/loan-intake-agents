"""The agent team as a LangGraph state graph.

The supervisor is a routing function (core/routing.py); each node is one specialist. Nodes talk
to the outside world only through ports. Case state holds plain JSON-like values; contracts
validate them on the way in and out of every node.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from loan_agents.contracts import (
    ApprovalRequest,
    CaseStatus,
    CrmDeal,
    CrmRefs,
    DealStage,
    DecisionAction,
    DocumentIssue,
    LoanRequest,
    ManagerDecision,
    QuestionTopic,
    Valuation,
    VehicleCertificate,
)
from loan_agents.core import contract as contract_rules
from loan_agents.core import documents as document_rules
from loan_agents.core import intake as intake_rules
from loan_agents.core import texts_uk as texts
from loan_agents.core.policy import Policy, value_application
from loan_agents.core.routing import RouteFacts, Step, TurnInput, next_step
from loan_agents.ports import AuditLog, CrmGateway, DocumentStore, Extractor, PriceBook

log = logging.getLogger(__name__)

_MAX_NOTE_CHARS = 300


class CaseState(TypedDict, total=False):
    case_id: str
    channel: str
    status: str
    turn_input: str
    inbox_text: str | None
    pending_documents: list[dict[str, str]]
    request: dict[str, Any]
    certificate: dict[str, Any] | None
    document_flags: list[str]
    valuation: dict[str, Any] | None
    crm: dict[str, str] | None
    contract_text: str | None
    contract_doc_id: str | None
    decision: dict[str, Any] | None
    notes_for_manager: Annotated[list[str], operator.add]
    # Every message sent to the client, append-only. The service sends the new tail each turn.
    transcript: Annotated[list[str], operator.add]


@dataclass(frozen=True)
class AgentDeps:
    extractor: Extractor
    prices: PriceBook
    crm: CrmGateway
    documents: DocumentStore
    audit: AuditLog
    policy: Policy
    contract_template: str
    today: Callable[[], date]


def _request(state: CaseState) -> LoanRequest:
    return LoanRequest.model_validate(state.get("request") or {})


def _certificate(state: CaseState) -> VehicleCertificate:
    return VehicleCertificate.model_validate(state["certificate"])


def _valuation(state: CaseState) -> Valuation:
    return Valuation.model_validate(state["valuation"])


def _document_flags(state: CaseState) -> list[DocumentIssue]:
    return [DocumentIssue(flag) for flag in state.get("document_flags") or []]


class _Team:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    @property
    def _year(self) -> int:
        return self.deps.today().year

    def supervisor(self, state: CaseState) -> str:
        valuation = state.get("valuation")
        facts = RouteFacts(
            status=CaseStatus(state.get("status", CaseStatus.COLLECTING)),
            turn_input=TurnInput(state.get("turn_input", TurnInput.START)),
            has_unread_text=bool(state.get("inbox_text")),
            has_unread_document=bool(state.get("pending_documents")),
            missing_fields=intake_rules.missing_fields(_request(state), self.deps.policy),
            has_certificate=state.get("certificate") is not None,
            has_valuation=valuation is not None,
            has_crm_deal=bool((state.get("crm") or {}).get("deal_id")),
            needs_contract=(
                valuation is not None
                and not valuation["flags"]
                and not state.get("document_flags")
                and not state.get("contract_text")
            ),
        )
        step = next_step(facts)
        log.info("case=%s step=supervisor next=%s", state.get("case_id"), step)
        return step.value

    async def intake(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        text = state["inbox_text"] or ""
        extraction = await self.deps.extractor.extract_request(text)
        result = intake_rules.merge_extraction(
            _request(state), extraction, self.deps.policy, self._year
        )
        replies = [
            texts.invalid_field(field, self.deps.policy, self._year)
            for field in result.rejected_fields
        ]
        notes = []
        topic = QuestionTopic(extraction.question_topic) if extraction.question_topic else None
        if topic is not None:
            replies.insert(0, texts.faq_answer(topic, self.deps.policy))
            if topic is QuestionTopic.OTHER:
                notes.append(" ".join(text.split())[:_MAX_NOTE_CHARS])
        await self.deps.audit.record(
            case_id,
            "intake",
            "extracted",
            {
                "accepted": list(result.accepted_fields),
                "rejected": list(result.rejected_fields),
                "question": topic.value if topic else None,
            },
        )
        return {
            "request": result.request.model_dump(),
            "inbox_text": None,
            "transcript": replies,
            "notes_for_manager": notes,
        }

    async def documents(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        latest = state["pending_documents"][-1]
        data, mime_type = await self.deps.documents.load(case_id, latest["doc_id"])
        reading = await self.deps.extractor.read_certificate(data, mime_type)
        check = document_rules.check_certificate(reading, _request(state), self._year)
        issues = [issue.value for issue in check.issues]
        if check.certificate is None:
            await self.deps.audit.record(case_id, "documents", "rejected", {"issues": issues})
            return {
                "pending_documents": [],
                "transcript": [texts.photo_issue(check.issues)],
            }
        await self.deps.audit.record(case_id, "documents", "accepted", {"flags": issues})
        # A new certificate invalidates everything computed from the previous one.
        return {
            "pending_documents": [],
            "certificate": check.certificate.model_dump(),
            "document_flags": issues,
            "valuation": None,
            "contract_text": None,
            "contract_doc_id": None,
            "transcript": [texts.DOCUMENT_ACCEPTED],
        }

    async def valuation(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        request = _request(state)
        certificate = _certificate(state)
        if request.amount_uah is None or request.term_months is None:
            raise ValueError(f"case {case_id}: valuation reached without amount and term")
        market_value = self.deps.prices.find_price_uah(
            certificate.make, certificate.model, certificate.year
        )
        valuation = value_application(
            amount_uah=request.amount_uah,
            term_months=request.term_months,
            car_year=certificate.year,
            market_value_uah=market_value,
            policy=self.deps.policy,
            current_year=self._year,
        )
        await self.deps.audit.record(
            case_id,
            "valuation",
            "done",
            {
                "market_value_uah": market_value,
                "loan_to_value": valuation.loan_to_value,
                "flags": [flag.value for flag in valuation.flags],
            },
        )
        return {"valuation": valuation.model_dump(mode="json")}

    async def crm(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        request = _request(state)
        certificate = _certificate(state)
        valuation = _valuation(state)
        document_flags = _document_flags(state)
        if request.phone is None or request.full_name is None:
            raise ValueError(f"case {case_id}: CRM step reached without name and phone")
        # Certificates print the owner as SURNAME NAME PATRONYMIC; use it when it is the client.
        owner_parts = [
            intake_rules.capitalize_name_token(token)
            for token in certificate.owner_full_name.split()
        ]
        if DocumentIssue.OWNER_MISMATCH in document_flags or len(owner_parts) < 2:
            first_name, last_name, middle_name = request.full_name, "", ""
        else:
            last_name, first_name = owner_parts[0], owner_parts[1]
            middle_name = " ".join(owner_parts[2:])
        deal = CrmDeal(
            case_id=case_id,
            title=texts.crm_deal_title(certificate, valuation.requested_uah),
            first_name=first_name,
            last_name=last_name,
            middle_name=middle_name,
            phone=request.phone,
            amount_uah=valuation.requested_uah,
            stage=DealStage.REVIEW,
            comment=texts.crm_summary(
                request=request,
                certificate=certificate,
                valuation=valuation,
                document_flags=document_flags,
                notes=state.get("notes_for_manager") or [],
                policy=self.deps.policy,
            ),
        )
        refs = await self.deps.crm.upsert_deal(deal)
        await self.deps.audit.record(case_id, "crm", "deal_saved", refs.model_dump())
        return {"crm": refs.model_dump()}

    async def contract(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        data = contract_rules.build_contract_data(
            case_id=case_id,
            request=_request(state),
            certificate=_certificate(state),
            valuation=_valuation(state),
            today=self.deps.today(),
        )
        text = contract_rules.render_contract(self.deps.contract_template, data)
        doc_id = await self.deps.documents.save_text(case_id, "contract-draft.md", text)
        refs = CrmRefs.model_validate(state["crm"])
        await self.deps.crm.add_comment(refs.deal_id, texts.crm_contract_comment(text))
        await self.deps.audit.record(case_id, "contract", "drafted", {"doc_id": doc_id})
        return {"contract_text": text, "contract_doc_id": doc_id}

    async def ask(self, state: CaseState) -> dict[str, Any]:
        missing = intake_rules.missing_fields(_request(state), self.deps.policy)
        replies = [texts.ask_field(missing[0], self.deps.policy, self._year)]
        if not state.get("transcript"):
            replies.insert(0, texts.GREETING)
        return {"transcript": replies}

    async def ask_document(self, state: CaseState) -> dict[str, Any]:
        return {"transcript": [texts.ASK_DOCUMENT]}

    async def submit(self, state: CaseState) -> dict[str, Any]:
        await self.deps.audit.record(state["case_id"], "submit", "awaiting_manager", {})
        return {"status": CaseStatus.AWAITING_MANAGER.value, "transcript": [texts.SUBMITTED]}

    def approval(self, state: CaseState) -> dict[str, Any]:
        # Re-runs from the top when the manager resumes it, so nothing here may have side effects.
        refs = CrmRefs.model_validate(state["crm"])
        request = ApprovalRequest(
            case_id=state["case_id"],
            card=texts.manager_card(
                case_id=state["case_id"],
                request=_request(state),
                certificate=_certificate(state),
                valuation=_valuation(state),
                document_flags=_document_flags(state),
                notes=state.get("notes_for_manager") or [],
                crm_deal_id=refs.deal_id,
                has_contract=bool(state.get("contract_text")),
                policy=self.deps.policy,
            ),
            contract_text=state.get("contract_text"),
        )
        resume_value = interrupt(request.model_dump())
        decision = ManagerDecision.model_validate(resume_value)
        return {"decision": decision.model_dump(mode="json")}

    async def finalize(self, state: CaseState) -> dict[str, Any]:
        case_id = state["case_id"]
        decision = ManagerDecision.model_validate(state["decision"])
        refs = CrmRefs.model_validate(state["crm"])
        update: dict[str, Any]
        if decision.action is DecisionAction.APPROVE:
            stage = DealStage.APPROVED
            update = {
                "status": CaseStatus.APPROVED.value,
                "transcript": [texts.DECISION_APPROVED],
            }
        elif decision.action is DecisionAction.REJECT:
            stage = DealStage.REJECTED
            update = {
                "status": CaseStatus.REJECTED.value,
                "transcript": [texts.DECISION_REJECTED],
            }
        else:
            stage = DealStage.NEW
            update = {
                "status": CaseStatus.COLLECTING.value,
                "certificate": None,
                "document_flags": [],
                "valuation": None,
                "contract_text": None,
                "contract_doc_id": None,
                "decision": None,
                "transcript": [texts.decision_need_documents(decision.comment)],
            }
        await self.deps.crm.set_stage(refs.deal_id, stage)
        await self.deps.crm.add_comment(refs.deal_id, texts.crm_decision_comment(decision))
        await self.deps.audit.record(
            case_id,
            "finalize",
            "decided",
            {"action": decision.action.value, "by": decision.manager},
        )
        return update

    async def closed(self, state: CaseState) -> dict[str, Any]:
        return {"transcript": [texts.CASE_CLOSED]}


_ROUTES: dict[str, str] = {step.value: step.value for step in Step if step is not Step.WAIT}
_ROUTES[Step.WAIT.value] = END


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    team = _Team(deps)
    graph = StateGraph(CaseState)
    graph.add_node("intake", team.intake)
    graph.add_node("documents", team.documents)
    graph.add_node("valuation", team.valuation)
    graph.add_node("crm", team.crm)
    graph.add_node("contract", team.contract)
    graph.add_node("ask", team.ask)
    graph.add_node("ask_document", team.ask_document)
    graph.add_node("submit", team.submit)
    graph.add_node("approval", team.approval)
    graph.add_node("finalize", team.finalize)
    graph.add_node("closed", team.closed)

    graph.add_conditional_edges(START, team.supervisor, _ROUTES)
    for worker in ("intake", "documents", "valuation", "crm", "contract"):
        graph.add_conditional_edges(worker, team.supervisor, _ROUTES)
    for reply_node in ("ask", "ask_document", "closed"):
        graph.add_edge(reply_node, END)
    graph.add_edge("submit", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
