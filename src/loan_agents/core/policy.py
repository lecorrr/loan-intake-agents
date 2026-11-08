"""The lending policy and the valuation maths."""

from __future__ import annotations

import math
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from loan_agents.contracts import REQUEST_FIELDS, PolicyFlag, Valuation


class IntRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="after")
    def _min_not_above_max(self) -> IntRange:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) is above max ({self.max})")
        return self

    def contains(self, value: int) -> bool:
        return self.min <= value <= self.max


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    amount_uah: IntRange
    term_months: IntRange
    max_loan_to_value: float = Field(gt=0, le=1)
    max_car_age_years: int = Field(ge=0, le=60)
    annual_rate_pct: float = Field(gt=0, le=200)
    required_fields: tuple[str, ...] = Field(min_length=1)

    @field_validator("term_months")
    @classmethod
    def _term_at_least_one_month(cls, value: IntRange) -> IntRange:
        if value.min < 1:
            raise ValueError("min must be at least 1 month")
        return value

    @field_validator("required_fields")
    @classmethod
    def _known_unique_fields(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        unknown = [name for name in value if name not in REQUEST_FIELDS]
        if unknown:
            raise ValueError(f"unknown fields {unknown}; allowed: {list(REQUEST_FIELDS)}")
        if len(set(value)) != len(value):
            raise ValueError("fields are listed more than once")
        return value


def monthly_payment_uah(amount_uah: int, term_months: int, annual_rate_pct: float) -> int:
    """Annuity payment, rounded up to a whole hryvnia."""
    if amount_uah <= 0 or term_months <= 0 or annual_rate_pct < 0:
        raise ValueError(
            f"amount {amount_uah}, term {term_months} and rate {annual_rate_pct} must be positive"
        )
    monthly_rate = annual_rate_pct / 100 / 12
    if monthly_rate == 0:
        payment = amount_uah / term_months
    else:
        payment = amount_uah * monthly_rate / (1 - (1 + monthly_rate) ** -term_months)
    return math.ceil(round(payment, 2))


def value_application(
    *,
    amount_uah: int,
    term_months: int,
    car_year: int,
    market_value_uah: int | None,
    policy: Policy,
    current_year: int,
) -> Valuation:
    """Compares the request with the policy. Flags go to the manager; nothing is decided here."""
    flags: list[PolicyFlag] = []
    if current_year - car_year > policy.max_car_age_years:
        flags.append(PolicyFlag.CAR_TOO_OLD)

    max_loan_uah: int | None = None
    loan_to_value: float | None = None
    if market_value_uah is None:
        flags.append(PolicyFlag.NO_MARKET_PRICE)
    else:
        if market_value_uah <= 0:
            raise ValueError(f"market value must be positive, got {market_value_uah}")
        # Decimal keeps 560000 * 0.6 at exactly 336000, which floats do not.
        limit = Decimal(market_value_uah) * Decimal(str(policy.max_loan_to_value))
        max_loan_uah = int(limit // 1000 * 1000)
        loan_to_value = round(amount_uah / market_value_uah, 4)
        if amount_uah > max_loan_uah:
            flags.append(PolicyFlag.ABOVE_MAX_LOAN_TO_VALUE)

    return Valuation(
        market_value_uah=market_value_uah,
        max_loan_uah=max_loan_uah,
        requested_uah=amount_uah,
        term_months=term_months,
        loan_to_value=loan_to_value,
        annual_rate_pct=policy.annual_rate_pct,
        monthly_payment_uah=monthly_payment_uah(amount_uah, term_months, policy.annual_rate_pct),
        flags=flags,
    )
