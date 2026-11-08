"""Filling the loan-agreement template. The template text is loaded by the caller."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from jinja2 import Environment, StrictUndefined

from loan_agents.contracts import LoanRequest, Valuation, VehicleCertificate
from loan_agents.core.texts_uk import format_pct, format_uah


def build_contract_data(
    *,
    case_id: str,
    request: LoanRequest,
    certificate: VehicleCertificate,
    valuation: Valuation,
    today: date,
) -> dict[str, str]:
    if valuation.market_value_uah is None or valuation.loan_to_value is None:
        raise ValueError(f"case {case_id}: a contract needs a market value")
    if request.full_name is None or request.phone is None:
        raise ValueError(f"case {case_id}: a contract needs the borrower's name and phone")
    return {
        "date": today.strftime("%d.%m.%Y"),
        "case_id": case_id,
        "borrower_name": request.full_name,
        "phone": request.phone,
        "amount": format_uah(valuation.requested_uah),
        "term_months": str(valuation.term_months),
        "rate": f"{valuation.annual_rate_pct:g}",
        "monthly_payment": format_uah(valuation.monthly_payment_uah),
        "make": certificate.make,
        "model": certificate.model,
        "year": str(certificate.year),
        "plate": certificate.plate_number,
        "vin": certificate.vin,
        "owner": certificate.owner_full_name,
        "market_value": format_uah(valuation.market_value_uah),
        "loan_to_value": format_pct(valuation.loan_to_value),
    }


def render_contract(template_text: str, data: Mapping[str, str]) -> str:
    """StrictUndefined turns a placeholder missing from the data into an error, not a blank."""
    environment = Environment(
        undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True
    )
    return environment.from_string(template_text).render(**data)
