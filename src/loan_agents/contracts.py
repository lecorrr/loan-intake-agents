"""The vocabulary shared by every module.

Units and formats, fixed here once:
- money in whole hryvnias (UAH), loan terms in months, years as calendar years;
- phones as +380XXXXXXXXX;
- car makes and models in upper-case Latin, as printed on Ukrainian registration
  certificates (SKODA, OCTAVIA); plates as AA1234BB with Latin letters.

Models named *Extraction and *Reading are raw model output. Nothing acts on them until
core/ has validated them into the strict models below.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

CASE_ID_PATTERN = re.compile(r"^(tg|web)-[A-Za-z0-9_-]{1,64}$")

REQUEST_FIELDS = (
    "full_name",
    "phone",
    "amount_uah",
    "term_months",
    "car_make",
    "car_model",
    "car_year",
    "region",
)


class Channel(StrEnum):
    TELEGRAM = "telegram"
    WEB = "web"


class CaseStatus(StrEnum):
    COLLECTING = "collecting"
    AWAITING_MANAGER = "awaiting_manager"
    APPROVED = "approved"
    REJECTED = "rejected"


class DecisionAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    NEED_DOCUMENTS = "need_documents"


class QuestionTopic(StrEnum):
    RATE = "rate"
    DOCUMENTS = "documents"
    TIMING = "timing"
    OTHER = "other"


class PolicyFlag(StrEnum):
    CAR_TOO_OLD = "car_too_old"
    NO_MARKET_PRICE = "no_market_price"
    ABOVE_MAX_LOAN_TO_VALUE = "above_max_loan_to_value"


class DocumentIssue(StrEnum):
    NOT_A_CERTIFICATE = "not_a_certificate"
    UNREADABLE = "unreadable"
    INVALID_VIN = "invalid_vin"
    INVALID_PLATE = "invalid_plate"
    MAKE_MISMATCH = "make_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    YEAR_MISMATCH = "year_mismatch"
    OWNER_MISMATCH = "owner_mismatch"

    @property
    def needs_new_photo(self) -> bool:
        """True when the client must resend the photo; False when only the manager must look."""
        return self in _PHOTO_ISSUES


_PHOTO_ISSUES = frozenset(
    {
        DocumentIssue.NOT_A_CERTIFICATE,
        DocumentIssue.UNREADABLE,
        DocumentIssue.INVALID_VIN,
        DocumentIssue.INVALID_PLATE,
    }
)


class DealStage(StrEnum):
    NEW = "new"
    REVIEW = "review"
    APPROVED = "approved"
    REJECTED = "rejected"


class LoanRequest(BaseModel):
    """What the client has told us so far. Every value here has passed validation."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = None
    phone: str | None = None
    amount_uah: int | None = None
    term_months: int | None = None
    car_make: str | None = None
    car_model: str | None = None
    car_year: int | None = None
    region: str | None = None


class RequestExtraction(BaseModel):
    """Raw model output for one client message."""

    model_config = ConfigDict(extra="ignore")

    full_name: str | None = Field(None, description="The client's own name, as written.")
    phone: str | None = Field(None, description="The client's phone number, as written.")
    amount_uah: int | None = Field(
        None, description="Requested loan amount in hryvnias, e.g. '200 тис' -> 200000."
    )
    term_months: int | None = Field(
        None, description="Loan term in months, e.g. 'рік' -> 12, 'два роки' -> 24."
    )
    car_make: str | None = Field(
        None, description="Car make in Latin script as on registration documents, e.g. SKODA."
    )
    car_model: str | None = Field(
        None, description="Car model in Latin script as on registration documents, e.g. OCTAVIA."
    )
    car_year: int | None = Field(None, description="The car's year of manufacture.")
    region: str | None = Field(None, description="Oblast or city the client names.")
    question_topic: Literal["rate", "documents", "timing", "other"] | None = Field(
        None,
        description="If the client asks a question: its topic. Otherwise null.",
    )


class CertificateReading(BaseModel):
    """Raw model output for one photo."""

    model_config = ConfigDict(extra="ignore")

    is_vehicle_certificate: bool = Field(
        description="True only if the image is a vehicle registration certificate."
    )
    plate_number: str | None = Field(None, description="Registration number, e.g. AA1234BB.")
    vin: str | None = Field(None, description="The 17-character vehicle identification number.")
    make: str | None = Field(None, description="Make, as printed.")
    model: str | None = Field(None, description="Model, as printed.")
    year: int | None = Field(None, description="Year of manufacture.")
    owner_full_name: str | None = Field(
        None, description="Owner's surname, name and patronymic, as printed."
    )


class VehicleCertificate(BaseModel):
    """A certificate whose fields passed validation."""

    model_config = ConfigDict(extra="forbid")

    plate_number: str
    vin: str
    make: str
    model: str
    year: int
    owner_full_name: str


class Valuation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    market_value_uah: int | None
    max_loan_uah: int | None
    requested_uah: int
    term_months: int
    loan_to_value: float | None
    annual_rate_pct: float
    monthly_payment_uah: int
    flags: list[PolicyFlag]

    @property
    def within_policy(self) -> bool:
        return not self.flags


class CrmRefs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_id: str
    deal_id: str
    # Where a manager opens the deal; None when the CRM has no web page for it.
    deal_url: str | None = None


class CrmDeal(BaseModel):
    """Everything the CRM needs to create or update one deal."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    title: str
    first_name: str
    last_name: str = ""
    middle_name: str = ""
    phone: str
    amount_uah: int
    stage: DealStage
    comment: str


class ManagerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    manager: str = Field(min_length=1, max_length=100)
    comment: str | None = Field(None, max_length=1000)


class ApprovalRequest(BaseModel):
    """What the manager sees when a case file is complete."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    card: str
    contract_text: str | None = None


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: CaseStatus
    replies: list[str]
    awaiting_manager: bool


class CaseView(BaseModel):
    """A case as the API shows it. Personal data is masked."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: CaseStatus
    awaiting_manager: bool
    request: dict[str, str | int | None]
    certificate: dict[str, str | int] | None
    valuation: Valuation | None
    crm: CrmRefs | None
    flags: list[str]
    transcript: list[str]
