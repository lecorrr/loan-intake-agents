"""Validating a certificate reading and cross-checking it with the application."""

from __future__ import annotations

from dataclasses import dataclass

from loan_agents.contracts import (
    CertificateReading,
    DocumentIssue,
    LoanRequest,
    VehicleCertificate,
)
from loan_agents.core.vehicles import (
    MIN_CAR_YEAR,
    is_valid_plate,
    is_valid_vin,
    normalize_make,
    normalize_model,
    normalize_plate,
    normalize_vin,
)


@dataclass(frozen=True)
class DocumentCheck:
    certificate: VehicleCertificate | None
    issues: tuple[DocumentIssue, ...]


def _name_tokens(name: str) -> set[str]:
    return {token.upper().replace("’", "'").replace("ʼ", "'") for token in name.split()}


def names_match(client_name: str, owner_name: str) -> bool:
    """The client's name must be part of the owner's; the client may leave out the patronymic."""
    client = _name_tokens(client_name)
    return len(client) >= 2 and client <= _name_tokens(owner_name)


def check_certificate(
    reading: CertificateReading,
    request: LoanRequest,
    current_year: int,
) -> DocumentCheck:
    if not reading.is_vehicle_certificate:
        return DocumentCheck(None, (DocumentIssue.NOT_A_CERTIFICATE,))

    raw = (
        reading.plate_number,
        reading.vin,
        reading.make,
        reading.model,
        reading.owner_full_name,
    )
    if any(value is None or not value.strip() for value in raw) or reading.year is None:
        return DocumentCheck(None, (DocumentIssue.UNREADABLE,))
    if not MIN_CAR_YEAR <= reading.year <= current_year:
        return DocumentCheck(None, (DocumentIssue.UNREADABLE,))

    plate = normalize_plate(reading.plate_number)
    vin = normalize_vin(reading.vin)
    photo_issues = []
    if not is_valid_vin(vin):
        photo_issues.append(DocumentIssue.INVALID_VIN)
    if not is_valid_plate(plate):
        photo_issues.append(DocumentIssue.INVALID_PLATE)
    if photo_issues:
        return DocumentCheck(None, tuple(photo_issues))

    certificate = VehicleCertificate(
        plate_number=plate,
        vin=vin,
        make=normalize_make(reading.make),
        model=normalize_model(reading.model),
        year=reading.year,
        owner_full_name=" ".join(reading.owner_full_name.upper().split()),
    )

    mismatches = []
    if request.car_make is not None and request.car_make != certificate.make:
        mismatches.append(DocumentIssue.MAKE_MISMATCH)
    if request.car_model is not None and request.car_model != certificate.model:
        mismatches.append(DocumentIssue.MODEL_MISMATCH)
    if request.car_year is not None and request.car_year != certificate.year:
        mismatches.append(DocumentIssue.YEAR_MISMATCH)
    if request.full_name is not None and not names_match(
        request.full_name, certificate.owner_full_name
    ):
        mismatches.append(DocumentIssue.OWNER_MISMATCH)
    return DocumentCheck(certificate, tuple(mismatches))
