from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from loan_agents.contracts import PolicyFlag
from loan_agents.core.policy import Policy, monthly_payment_uah, value_application
from loan_agents.settings import ConfigError, load_policy

POLICY_FIELDS = {
    "amount_uah": {"min": 20000, "max": 2000000},
    "term_months": {"min": 3, "max": 36},
    "max_loan_to_value": 0.6,
    "max_car_age_years": 20,
    "annual_rate_pct": 18.0,
    "required_fields": ["full_name", "phone"],
}


def test_annuity_payment_matches_the_formula() -> None:
    # 200 000 UAH, 12 months, 18% a year: 1.5% a month.
    assert monthly_payment_uah(200_000, 12, 18.0) == 18_336


def test_zero_rate_divides_evenly() -> None:
    assert monthly_payment_uah(120_000, 12, 0.0) == 10_000


@pytest.mark.parametrize(
    ("amount", "term", "rate"), [(0, 12, 18.0), (1000, 0, 18.0), (1000, 12, -1)]
)
def test_payment_rejects_degenerate_inputs(amount: int, term: int, rate: float) -> None:
    with pytest.raises(ValueError):
        monthly_payment_uah(amount, term, rate)


def test_max_loan_is_exact_for_decimal_ratios(policy: Policy) -> None:
    valuation = value_application(
        amount_uah=200_000,
        term_months=12,
        car_year=2018,
        market_value_uah=560_000,
        policy=policy,
        current_year=2026,
    )
    assert valuation.max_loan_uah == 336_000
    assert valuation.loan_to_value == pytest.approx(0.3571, abs=1e-4)
    assert valuation.flags == []
    assert valuation.within_policy


@pytest.mark.parametrize(
    ("amount", "car_year", "market_value", "expected"),
    [
        (400_000, 2018, 560_000, [PolicyFlag.ABOVE_MAX_LOAN_TO_VALUE]),
        (100_000, 2004, 300_000, [PolicyFlag.CAR_TOO_OLD]),
        (100_000, 2018, None, [PolicyFlag.NO_MARKET_PRICE]),
        (100_000, 2004, None, [PolicyFlag.CAR_TOO_OLD, PolicyFlag.NO_MARKET_PRICE]),
    ],
)
def test_policy_flags(
    policy: Policy,
    amount: int,
    car_year: int,
    market_value: int | None,
    expected: list[PolicyFlag],
) -> None:
    valuation = value_application(
        amount_uah=amount,
        term_months=12,
        car_year=car_year,
        market_value_uah=market_value,
        policy=policy,
        current_year=2026,
    )
    assert valuation.flags == expected


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"amount_uah": {"min": 5, "max": 1}}, "above max"),
        ({"term_months": {"min": 0, "max": 12}}, "at least 1 month"),
        ({"max_loan_to_value": 1.5}, "less than or equal to 1"),
        ({"required_fields": ["full_name", "salary"]}, "unknown fields"),
        ({"required_fields": ["phone", "phone"]}, "more than once"),
        ({"bonus": 1}, "Extra inputs"),
    ],
)
def test_policy_rejects_invalid_values(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Policy.model_validate(POLICY_FIELDS | override)


def test_load_policy_names_the_file_and_the_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("amount_uah: {min: 1, max: 2}\nmax_ltv: 0.5\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_policy(path)
    assert str(path) in str(caught.value)
    assert "max_ltv" in str(caught.value)


def test_load_policy_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_policy(tmp_path / "missing.yaml")
