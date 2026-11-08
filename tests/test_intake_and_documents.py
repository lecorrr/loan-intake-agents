from __future__ import annotations

import pytest

from loan_agents.contracts import CertificateReading, DocumentIssue, LoanRequest, RequestExtraction
from loan_agents.core.documents import check_certificate, names_match
from loan_agents.core.intake import (
    merge_extraction,
    missing_fields,
    normalize_full_name,
    normalize_phone,
)
from loan_agents.core.policy import Policy

GOOD_READING = {
    "is_vehicle_certificate": True,
    "plate_number": "AA1234BB",
    "vin": "TMBJJ7NE8J0123456",
    "make": "SKODA",
    "model": "OCTAVIA",
    "year": 2018,
    "owner_full_name": "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ",
}
MATCHING_REQUEST = LoanRequest(
    full_name="Тарас Шевченко", car_make="SKODA", car_model="OCTAVIA", car_year=2018
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("067 123 45 67", "+380671234567"),
        ("+38 (067) 123-45-67", "+380671234567"),
        ("380671234567", "+380671234567"),
        ("12345", None),
        ("+44 20 7946 0958", None),
        ("", None),
    ],
)
def test_normalize_phone(raw: str, expected: str | None) -> None:
    assert normalize_phone(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("тарас шевченко", "Тарас Шевченко"),
        ("мар'яна петренко-коваль", "Мар'яна Петренко-Коваль"),
        ("Тарас", None),
        ("Robert'); DROP TABLE", None),
        ("один два три чотири пять", None),
    ],
)
def test_normalize_full_name(raw: str, expected: str | None) -> None:
    assert normalize_full_name(raw) == expected


def test_merge_accepts_valid_values_and_normalises_car_names(policy: Policy) -> None:
    extraction = RequestExtraction(
        full_name="тарас шевченко",
        phone="0671234567",
        amount_uah=200_000,
        term_months=12,
        car_make="Шкода",
        car_model="Октавія",
        car_year=2018,
    )
    result = merge_extraction(LoanRequest(), extraction, policy, current_year=2026)
    assert result.rejected_fields == ()
    assert result.request.car_make == "SKODA"
    assert result.request.car_model == "OCTAVIA"
    assert result.request.phone == "+380671234567"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount_uah", 5_000),
        ("amount_uah", 10_000_000),
        ("term_months", 0),
        ("term_months", 120),
        ("car_year", 2031),
        ("car_year", 1900),
        ("phone", "12"),
        ("full_name", "X"),
    ],
)
def test_merge_rejects_out_of_range_values(policy: Policy, field: str, value: object) -> None:
    current = LoanRequest(amount_uah=100_000)
    result = merge_extraction(current, RequestExtraction(**{field: value}), policy, 2026)
    assert result.rejected_fields == (field,)
    assert result.request == current


def test_merge_does_not_change_its_input(policy: Policy) -> None:
    current = LoanRequest()
    merge_extraction(current, RequestExtraction(amount_uah=100_000), policy, 2026)
    assert current.amount_uah is None


def test_missing_fields_follow_the_policy_order(policy: Policy) -> None:
    request = LoanRequest(phone="+380671234567", car_make="SKODA")
    assert missing_fields(request, policy) == (
        "full_name",
        "amount_uah",
        "term_months",
        "car_model",
        "car_year",
    )


def test_a_matching_certificate_is_accepted() -> None:
    check = check_certificate(CertificateReading(**GOOD_READING), MATCHING_REQUEST, 2026)
    assert check.issues == ()
    assert check.certificate is not None
    assert check.certificate.plate_number == "AA1234BB"


def test_cyrillic_letters_in_the_plate_are_normalised() -> None:
    reading = CertificateReading(**(GOOD_READING | {"plate_number": "АА 1234 ВВ"}))
    check = check_certificate(reading, MATCHING_REQUEST, 2026)
    assert check.certificate is not None
    assert check.certificate.plate_number == "AA1234BB"


@pytest.mark.parametrize(
    ("override", "issue"),
    [
        ({"is_vehicle_certificate": False}, DocumentIssue.NOT_A_CERTIFICATE),
        ({"vin": None}, DocumentIssue.UNREADABLE),
        ({"owner_full_name": "  "}, DocumentIssue.UNREADABLE),
        ({"year": 2035}, DocumentIssue.UNREADABLE),
        ({"vin": "TMBJJ7NE8J012345O"}, DocumentIssue.INVALID_VIN),
        ({"vin": "TMBJJ7NE8J01234"}, DocumentIssue.INVALID_VIN),
        ({"plate_number": "A1234BB"}, DocumentIssue.INVALID_PLATE),
    ],
)
def test_unusable_photos_ask_for_a_new_one(
    override: dict[str, object], issue: DocumentIssue
) -> None:
    check = check_certificate(
        CertificateReading(**(GOOD_READING | override)), MATCHING_REQUEST, 2026
    )
    assert check.certificate is None
    assert check.issues == (issue,)
    assert issue.needs_new_photo


@pytest.mark.parametrize(
    ("request_override", "issue"),
    [
        ({"car_make": "TOYOTA"}, DocumentIssue.MAKE_MISMATCH),
        ({"car_model": "FABIA"}, DocumentIssue.MODEL_MISMATCH),
        ({"car_year": 2017}, DocumentIssue.YEAR_MISMATCH),
        ({"full_name": "Оксана Шевченко"}, DocumentIssue.OWNER_MISMATCH),
    ],
)
def test_mismatches_go_to_the_manager(
    request_override: dict[str, object], issue: DocumentIssue
) -> None:
    request = MATCHING_REQUEST.model_copy(update=request_override)
    check = check_certificate(CertificateReading(**GOOD_READING), request, 2026)
    assert check.certificate is not None
    assert check.issues == (issue,)
    assert not issue.needs_new_photo


@pytest.mark.parametrize(
    ("client", "owner", "expected"),
    [
        ("Тарас Шевченко", "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ", True),
        ("Шевченко Тарас Григорович", "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ", True),
        ("Мар’яна Бойко", "БОЙКО МАР'ЯНА ІВАНІВНА", True),
        ("Оксана Шевченко", "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ", False),
        ("Тарас", "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ", False),
    ],
)
def test_names_match(client: str, owner: str, expected: bool) -> None:
    assert names_match(client, owner) is expected
