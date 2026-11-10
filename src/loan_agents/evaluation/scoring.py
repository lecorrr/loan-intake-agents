"""Scoring: the model's answers field by field, and each case's outcome and guardrails."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from loan_agents.contracts import (
    CaseStatus,
    CaseView,
    CertificateReading,
    DealStage,
    DecisionAction,
    DocumentIssue,
    LoanRequest,
    ManagerDecision,
    RequestExtraction,
    TurnResult,
)
from loan_agents.core import intake, masking
from loan_agents.core import texts_uk as texts
from loan_agents.core.vehicles import (
    normalize_client_make,
    normalize_client_model,
    normalize_make,
    normalize_model,
    normalize_plate,
    normalize_vin,
)
from loan_agents.evaluation.cases import Certificate, Expect, Outcome, Step
from loan_agents.fakes.crm import FakeDeal

# The question topic is scored like a field: recognising the question is part of reading a message.
SCORED_FIELDS = (
    "full_name",
    "phone",
    "amount_uah",
    "term_months",
    "car_make",
    "car_model",
    "car_year",
    "question",
)
READ_FIELDS = tuple(CertificateReading.model_fields)


@dataclass
class FieldScore:
    right: int = 0
    wrong: int = 0
    missed: int = 0
    # Values the model gave that the message or photo does not hold.
    extra: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def expected(self) -> int:
        return self.right + self.wrong + self.missed

    def add(self, other: FieldScore) -> None:
        self.right += other.right
        self.wrong += other.wrong
        self.missed += other.missed
        self.extra += other.extra
        self.notes += other.notes

    def count(self, label: str, name: str, want: object, have: object, same: bool) -> None:
        if want is None and have is None:
            return
        if same:
            self.right += 1
        elif want is None:
            self.extra += 1
            self.notes.append(f"{label}: {name} {have!r}, which is not there")
        elif have is None:
            self.missed += 1
            self.notes.append(f"{label}: {name} missed, expected {want!r}")
        else:
            self.wrong += 1
            self.notes.append(f"{label}: {name} {have!r}, expected {want!r}")


def _name_key(name: str) -> str:
    normalized = intake.normalize_full_name(name) or name
    return normalized.casefold().replace("’", "'").replace("ʼ", "'")


def _comparable(name: str, value: object) -> object:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    text = str(value)
    if name == "full_name":
        return _name_key(text)
    if name == "phone":
        return intake.normalize_phone(text) or text
    if name == "car_make":
        return normalize_client_make(text)
    if name == "car_model":
        return normalize_client_model(text)
    return value


def score_message(label: str, expected: Expect, got: RequestExtraction) -> FieldScore:
    score = FieldScore()
    for name in SCORED_FIELDS:
        want: object
        have: object
        if name == "question":
            want = expected.question.value if expected.question else None
            have = got.question_topic
        else:
            want, have = getattr(expected, name), getattr(got, name)
        same = _comparable(name, want) == _comparable(name, have)
        score.count(label, name, want, have, same)
    return score


def reading_field_matches(name: str, truth: object, read: object) -> bool:
    """Whether a certificate field was read right, ignoring spacing and letter case."""
    if truth is None or read is None:
        return truth == read
    normalizers = {
        "plate_number": normalize_plate,
        "vin": normalize_vin,
        "make": normalize_make,
        "model": normalize_model,
        "owner_full_name": lambda value: " ".join(value.upper().split()),
    }
    if name in normalizers:
        return normalizers[name](str(truth)) == normalizers[name](str(read))
    return truth == read


def score_reading(label: str, truth: CertificateReading, got: CertificateReading) -> FieldScore:
    score = FieldScore()
    for name in READ_FIELDS:
        want, have = getattr(truth, name), getattr(got, name)
        score.count(label, name, want, have, reading_field_matches(name, want, have))
    return score


@dataclass
class CaseResult:
    id: str
    about: str
    session_id: str
    problems: list[str] = field(default_factory=list)
    # Broken guardrails: these must never happen, whatever the model gets wrong.
    breaches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    fields: FieldScore = field(default_factory=FieldScore)
    reading: FieldScore = field(default_factory=FieldScore)
    # (turn kind, milliseconds): start, message, photo or decision.
    turns: list[tuple[str, float]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.problems and not self.breaches


_PHOTO_REJECTIONS = frozenset(
    texts.photo_issue([issue]) for issue in DocumentIssue if issue.needs_new_photo
)
_STATUS_AFTER = {
    DecisionAction.APPROVE: CaseStatus.APPROVED,
    DecisionAction.REJECT: CaseStatus.REJECTED,
    DecisionAction.NEED_DOCUMENTS: CaseStatus.COLLECTING,
}
_STAGE_AFTER = {
    DecisionAction.APPROVE: DealStage.APPROVED,
    DecisionAction.REJECT: DealStage.REJECTED,
    DecisionAction.NEED_DOCUMENTS: DealStage.NEW,
}


def _short(replies: Sequence[str]) -> str:
    said = " / ".join(replies) or "nothing"
    return said if len(said) <= 120 else said[:117] + "..."


def check_photo_reply(result: CaseResult, label: str, step: Step, replies: list[str]) -> None:
    accepted = texts.DOCUMENT_ACCEPTED in replies
    rejection = next((reply for reply in replies if reply in _PHOTO_REJECTIONS), None)
    if step.rejected is None:
        if not accepted:
            result.problems.append(
                f"{label}: the photo was not accepted; the bot said: {_short(replies)}"
            )
        return
    if accepted or rejection is None:
        result.problems.append(
            f"{label}: the photo should be turned down ({step.rejected.value}); "
            f"the bot said: {_short(replies)}"
        )
    elif rejection != texts.photo_issue([step.rejected]):
        # Any reason makes the client resend the photo; the reason itself is the model's call.
        result.notes.append(f"{label}: photo turned down for another reason: {_short([rejection])}")


def check_manager(
    result: CaseResult, outcome: Outcome, cards: int, contract: str | None, view: CaseView
) -> bool:
    """Checks what the manager got; False when the case did not get as far as it should."""
    if cards > 1:
        result.breaches.append(f"the manager got {cards} cards for one case")
    reached = cards > 0
    if reached != outcome.manager:
        where = "reached the manager" if reached else "did not reach the manager"
        result.problems.append(f"{where}; last reply: {_short(view.transcript[-1:])}")
        return False
    if not reached:
        return True
    if view.certificate is None:
        result.breaches.append("reached the manager without a certificate")
    document_flags = sorted(view.flags)
    policy_flags = sorted(flag.value for flag in view.valuation.flags) if view.valuation else []
    expected_documents = sorted(flag.value for flag in outcome.document_flags)
    expected_policy = sorted(flag.value for flag in outcome.policy_flags)
    if document_flags != expected_documents:
        result.problems.append(
            f"document flags {document_flags or 'none'}, expected {expected_documents or 'none'}"
        )
    if policy_flags != expected_policy:
        result.problems.append(
            f"policy flags {policy_flags or 'none'}, expected {expected_policy or 'none'}"
        )
    drafted = contract is not None
    if drafted != outcome.contract:
        result.problems.append("a contract was drafted" if drafted else "no contract was drafted")
    if drafted and (document_flags or policy_flags):
        result.breaches.append("a contract was drafted for a flagged case")
    return True


def _without_region(values: dict[str, object]) -> dict[str, object]:
    return {name: value for name, value in values.items() if name != "region"}


def check_data(
    result: CaseResult,
    expected_request: LoanRequest,
    certificate: Certificate | None,
    view: CaseView,
) -> None:
    """Compares what the case holds with what the client gave, in the API's masked form."""
    want = _without_region(masking.masked_request(expected_request))
    have = _without_region(view.request)
    for name, value in want.items():
        if have.get(name) != value:
            result.problems.append(f"the case holds {name} {have.get(name)!r}, expected {value!r}")
    if certificate is None or view.certificate is None:
        return
    for name, value in masking.masked_certificate(certificate.printed()).items():
        if view.certificate.get(name) != value:
            result.problems.append(
                f"the case holds certificate {name} {view.certificate.get(name)!r}, "
                f"expected {value!r}"
            )


def check_decision(
    result: CaseResult, decision: ManagerDecision, decided: TurnResult, deal: FakeDeal | None
) -> None:
    action = decision.action
    if decided.status is not _STATUS_AFTER[action]:
        result.problems.append(
            f"status {decided.status.value} after {action.value}, "
            f"expected {_STATUS_AFTER[action].value}"
        )
    stage = deal.stage.value if deal else "no deal"
    if stage != _STAGE_AFTER[action].value:
        result.problems.append(f"CRM stage {stage}, expected {_STAGE_AFTER[action].value}")
    told = {
        DecisionAction.APPROVE: texts.DECISION_APPROVED,
        DecisionAction.REJECT: texts.DECISION_REJECTED,
        DecisionAction.NEED_DOCUMENTS: texts.decision_need_documents(decision.comment),
    }[action]
    if told not in decided.replies:
        result.problems.append(
            f"the client was not told the decision; the bot said: {_short(decided.replies)}"
        )


def check_no_early_decision(result: CaseResult, told: list[str], comment: str | None) -> None:
    decisions = {
        texts.DECISION_APPROVED,
        texts.DECISION_REJECTED,
        texts.decision_need_documents(None),
        texts.decision_need_documents(comment),
    }
    if any(reply in decisions for reply in told):
        result.breaches.append("the client was told a decision before the manager made one")
