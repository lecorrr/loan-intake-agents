"""Turning what the model extracted from a message into validated request fields."""

from __future__ import annotations

import re
from dataclasses import dataclass

from loan_agents.contracts import LoanRequest, RequestExtraction
from loan_agents.core.policy import Policy
from loan_agents.core.vehicles import MIN_CAR_YEAR, normalize_make, normalize_model

_NAME_TOKEN = re.compile(r"^[A-Za-zА-ЩЬЮЯҐЄІЇа-щьюяґєії][A-Za-zА-ЩЬЮЯҐЄІЇа-щьюяґєії'’ʼ-]{1,39}$")
_MAX_CAR_NAME_CHARS = 40
_MAX_REGION_CHARS = 60


@dataclass(frozen=True)
class IntakeResult:
    request: LoanRequest
    accepted_fields: tuple[str, ...]
    rejected_fields: tuple[str, ...]


def normalize_phone(raw: str) -> str | None:
    """Ukrainian mobile or landline number as +380XXXXXXXXX, or None if it isn't one."""
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 10 and digits.startswith("0"):
        digits = "38" + digits
    if len(digits) == 12 and digits.startswith("380"):
        return "+" + digits
    return None


def capitalize_name_token(token: str) -> str:
    """'ШЕВЧЕНКО' -> 'Шевченко', "мар'яна" -> "Мар'яна" (str.title() would give "Мар'Яна")."""
    return "-".join(part[:1].upper() + part[1:].lower() for part in token.split("-"))


def normalize_full_name(raw: str) -> str | None:
    tokens = raw.split()
    if not 2 <= len(tokens) <= 4:
        return None
    if not all(_NAME_TOKEN.match(token) for token in tokens):
        return None
    return " ".join(capitalize_name_token(token) for token in tokens)


def _normalize_car_name(raw: str, normalize) -> str | None:
    value = normalize(raw)
    if not value or len(value) > _MAX_CAR_NAME_CHARS:
        return None
    return value


def merge_extraction(
    current: LoanRequest,
    extraction: RequestExtraction,
    policy: Policy,
    current_year: int,
) -> IntakeResult:
    """Validates every field the model returned. Invalid values are dropped and reported."""
    updates: dict[str, object] = {}
    rejected: list[str] = []

    def accept(field: str, value: object | None) -> None:
        if value is None:
            rejected.append(field)
        else:
            updates[field] = value

    if extraction.full_name is not None:
        accept("full_name", normalize_full_name(extraction.full_name))
    if extraction.phone is not None:
        accept("phone", normalize_phone(extraction.phone))
    if extraction.amount_uah is not None:
        amount = extraction.amount_uah
        accept("amount_uah", amount if policy.amount_uah.contains(amount) else None)
    if extraction.term_months is not None:
        term = extraction.term_months
        accept("term_months", term if policy.term_months.contains(term) else None)
    if extraction.car_make is not None:
        accept("car_make", _normalize_car_name(extraction.car_make, normalize_make))
    if extraction.car_model is not None:
        accept("car_model", _normalize_car_name(extraction.car_model, normalize_model))
    if extraction.car_year is not None:
        year = extraction.car_year
        accept("car_year", year if MIN_CAR_YEAR <= year <= current_year else None)
    if extraction.region is not None:
        region = " ".join(extraction.region.split())
        accept("region", region if 2 <= len(region) <= _MAX_REGION_CHARS else None)

    return IntakeResult(
        request=current.model_copy(update=updates),
        accepted_fields=tuple(updates),
        rejected_fields=tuple(rejected),
    )


def missing_fields(request: LoanRequest, policy: Policy) -> tuple[str, ...]:
    return tuple(name for name in policy.required_fields if getattr(request, name) is None)
