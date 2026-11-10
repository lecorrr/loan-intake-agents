"""The case file: what each scripted application says, sends, and must lead to."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from loan_agents.contracts import (
    DecisionAction,
    DocumentIssue,
    PolicyFlag,
    QuestionTopic,
    RequestExtraction,
    VehicleCertificate,
)
from loan_agents.core import intake
from loan_agents.core.vehicles import is_valid_plate, is_valid_vin, normalize_make, normalize_model
from loan_agents.samples import Sample
from loan_agents.settings import PROJECT_ROOT, ConfigError, describe_validation_error

CASES_FILE = PROJECT_ROOT / "evals" / "cases.yaml"


class Expect(BaseModel):
    """The fields one client message holds, in the forms the app stores them."""

    model_config = ConfigDict(extra="forbid")

    full_name: str | None = None
    phone: str | None = None
    amount_uah: int | None = None
    term_months: int | None = None
    car_make: str | None = None
    car_model: str | None = None
    car_year: int | None = None
    question: QuestionTopic | None = None

    @model_validator(mode="after")
    def _stored_forms(self) -> Expect:
        stored = {
            "full_name": (self.full_name, intake.normalize_full_name),
            "phone": (self.phone, intake.normalize_phone),
            "car_make": (self.car_make, normalize_make),
            "car_model": (self.car_model, normalize_model),
        }
        for name, (value, normalize) in stored.items():
            if value is not None and normalize(value) != value:
                raise ValueError(f"{name} {value!r} should be written as {normalize(value)!r}")
        return self

    def as_extraction(self) -> RequestExtraction:
        values = self.model_dump(exclude={"question"})
        topic = self.question.value if self.question else None
        return RequestExtraction(**values, question_topic=topic)


class Certificate(BaseModel):
    """A certificate photo to draw: the printed fields, then how the photo looks."""

    model_config = ConfigDict(extra="forbid")

    plate_number: str
    vin: str
    make: str
    model: str
    year: int
    owner_full_name: str
    colour: str = "СІРИЙ"
    stamp: Literal["corner", "across"] = "corner"
    blur_px: int = Field(0, ge=0, le=20)

    @model_validator(mode="after")
    def _printable(self) -> Certificate:
        problems = []
        if not is_valid_plate(self.plate_number):
            problems.append(f"plate {self.plate_number!r} is not like AA1234BB")
        if not is_valid_vin(self.vin):
            problems.append(f"VIN {self.vin!r} is not 17 characters without I, O and Q")
        if normalize_make(self.make) != self.make or normalize_model(self.model) != self.model:
            problems.append(f"{self.make} {self.model}: write make and model as printed")
        if self.owner_full_name != " ".join(self.owner_full_name.upper().split()):
            problems.append(f"owner {self.owner_full_name!r}: write it in capitals, as printed")
        if problems:
            raise ValueError("; ".join(problems))
        return self

    def printed(self) -> VehicleCertificate:
        return VehicleCertificate.model_validate(
            self.model_dump(include=set(VehicleCertificate.model_fields))
        )


class Photo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    certificate: Certificate


class Step(BaseModel):
    """One client turn: a message, or a photo with an optional caption."""

    model_config = ConfigDict(extra="forbid")

    say: str | None = None
    send: Literal["receipt"] | Photo | None = None
    caption: str | None = None
    # What the message or the caption holds.
    expect: Expect = Expect()
    # A photo the bot must turn down, and why.
    rejected: DocumentIssue | None = None

    @model_validator(mode="after")
    def _one_turn(self) -> Step:
        if (self.say is None) == (self.send is None):
            raise ValueError("a step either says something or sends a photo")
        if self.say is not None and (self.caption is not None or self.rejected is not None):
            raise ValueError("caption and rejected go with send")
        if self.send is not None and self.caption is None and self.expect != Expect():
            raise ValueError("expect goes with say, or with send when there is a caption")
        if self.send == "receipt" and self.rejected is not DocumentIssue.NOT_A_CERTIFICATE:
            raise ValueError("a receipt must be rejected as not_a_certificate")
        if self.rejected is not None and not self.rejected.needs_new_photo:
            raise ValueError(f"{self.rejected.value} does not make the bot turn a photo down")
        return self

    @property
    def text(self) -> str | None:
        """What the client wrote in this step, if anything."""
        return self.say if self.say is not None else self.caption


class Outcome(BaseModel):
    """What the manager must receive."""

    model_config = ConfigDict(extra="forbid")

    manager: bool
    contract: bool = False
    document_flags: list[DocumentIssue] = []
    policy_flags: list[PolicyFlag] = []

    @model_validator(mode="after")
    def _consistent(self) -> Outcome:
        flagged = bool(self.document_flags or self.policy_flags)
        if self.contract and flagged:
            raise ValueError("a contract is drafted only when nothing is flagged")
        if not self.manager and (self.contract or flagged):
            raise ValueError("flags and the contract go to the manager; set manager: true")
        return self


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9-]{1,20}$")
    about: str = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)
    outcome: Outcome
    decision: DecisionAction | None = None
    comment: str | None = Field(None, max_length=1000)

    @model_validator(mode="after")
    def _decision_for_the_manager(self) -> EvalCase:
        if (self.decision is not None) != self.outcome.manager:
            raise ValueError(
                "a case that reaches the manager needs a decision, and only such a case"
            )
        if self.comment is not None and self.decision is None:
            raise ValueError("comment goes with decision")
        return self


class CaseFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The date the cases are judged on, so car ages do not change from year to year.
    today: date
    cases: list[EvalCase] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_ids(cls, cases: list[EvalCase]) -> list[EvalCase]:
        ids = [case.id for case in cases]
        repeated = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
        if repeated:
            raise ValueError(f"case ids used more than once: {repeated}")
        return cases


def load_cases(path: Path) -> CaseFile:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: not found") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from error
    try:
        return CaseFile.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(describe_validation_error(str(path), error)) from error


def photo_sample(case_id: str, step_number: int, send: Photo | Literal["receipt"]) -> Sample:
    name = f"{case_id}-{step_number}.png"
    if send == "receipt":
        return Sample(name, {"is_vehicle_certificate": False})
    certificate = send.certificate
    return Sample(
        name,
        {"is_vehicle_certificate": True} | certificate.printed().model_dump(),
        colour=certificate.colour,
        stamp=certificate.stamp,
        blur_px=certificate.blur_px,
    )
