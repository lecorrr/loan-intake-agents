"""Masking personal data for logs, the audit trail, the API and the manager's Telegram card.

Full values live only in the CRM, which is the system of record.
"""

from __future__ import annotations

from loan_agents.contracts import LoanRequest, VehicleCertificate


def mask_phone(phone: str) -> str:
    if len(phone) < 10:
        return "***"
    return f"{phone[:6]}***{phone[-4:]}"


def mask_vin(vin: str) -> str:
    if len(vin) < 8:
        return "***"
    return f"{vin[:3]}{'*' * (len(vin) - 7)}{vin[-4:]}"


def mask_plate(plate: str) -> str:
    if len(plate) < 5:
        return "***"
    return f"{plate[:2]}{'*' * (len(plate) - 4)}{plate[-2:]}"


def mask_name(name: str) -> str:
    return " ".join(f"{token[0]}." for token in name.split() if token)


def masked_request(request: LoanRequest) -> dict[str, str | int | None]:
    values = request.model_dump()
    if request.full_name:
        values["full_name"] = mask_name(request.full_name)
    if request.phone:
        values["phone"] = mask_phone(request.phone)
    return values


def masked_certificate(certificate: VehicleCertificate) -> dict[str, str | int]:
    values = certificate.model_dump()
    values["vin"] = mask_vin(certificate.vin)
    values["plate_number"] = mask_plate(certificate.plate_number)
    values["owner_full_name"] = mask_name(certificate.owner_full_name)
    return values
