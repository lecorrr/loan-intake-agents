from __future__ import annotations

import csv

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
from loan_agents.core.vehicles import KNOWN_MAKES, KNOWN_MODELS
from loan_agents.settings import PROJECT_ROOT

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
    ("make", "model", "expected"),
    [
        ("NISSAN", "KASHKAI", ("NISSAN", "QASHQAI")),
        ("Nissan", "Kashkay", ("NISSAN", "QASHQAI")),
        ("NISAN", "QASHQAI", ("NISSAN", "QASHQAI")),
        ("SHKODA", "OKTAVIYA", ("SKODA", "OCTAVIA")),
        ("Skoda", "Oktaviia", ("SKODA", "OCTAVIA")),
        ("TOIOTA", "KAMRI", ("TOYOTA", "CAMRY")),
        ("Toyota", "RAV 4", ("TOYOTA", "RAV4")),
        ("KHIUNDAI", "TUKSON", ("HYUNDAI", "TUCSON")),
        ("FORD", "FOKUS", ("FORD", "FOCUS")),
        ("VOLKSWAGEN", "PASAT", ("VOLKSWAGEN", "PASSAT")),
        ("BMV", "X5", ("BMW", "X5")),
    ],
)
def test_phonetic_spellings_of_known_car_names_are_recognised(
    policy: Policy, make: str, model: str, expected: tuple[str, str]
) -> None:
    extraction = RequestExtraction(car_make=make, car_model=model)
    result = merge_extraction(LoanRequest(), extraction, policy, current_year=2026)
    assert (result.request.car_make, result.request.car_model) == expected


@pytest.mark.parametrize(
    "model",
    [
        "X6",
        "CX-5",
        "TUCSON N",
        "OCTAVIA SCOUT",
        "PASSAT CC",
        "KAMIQ",
        "CIVIC",
        "ЛАНОС",
        pytest.param("X5 М", id="X5 with a Cyrillic M"),
    ],
)
def test_other_car_models_are_kept_as_written(policy: Policy, model: str) -> None:
    extraction = RequestExtraction(car_model=model)
    result = merge_extraction(LoanRequest(), extraction, policy, current_year=2026)
    assert result.request.car_model == model


def test_every_car_in_price_table_has_known_spellings() -> None:
    with (PROJECT_ROOT / "config" / "market_prices.csv").open(encoding="utf-8") as handle:
        cars = {(row["make"], row["model"]) for row in csv.DictReader(handle)}
    unknown = [
        f"{make} {model}"
        for make, model in sorted(cars)
        if make not in KNOWN_MAKES or model not in KNOWN_MODELS
    ]
    assert unknown == [], "add their Ukrainian spellings to core/vehicles.py"


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


def test_missing_fields_follow_policy_order(policy: Policy) -> None:
    request = LoanRequest(phone="+380671234567", car_make="SKODA")
    assert missing_fields(request, policy) == (
        "full_name",
        "amount_uah",
        "term_months",
        "car_model",
        "car_year",
    )


def test_matching_certificate_is_accepted() -> None:
    check = check_certificate(CertificateReading(**GOOD_READING), MATCHING_REQUEST, 2026)
    assert check.issues == ()
    assert check.certificate is not None
    assert check.certificate.plate_number == "AA1234BB"


def test_certificate_is_read_as_printed_not_matched_by_sound() -> None:
    # A misread certificate reaches the manager as a mismatch; it is never guessed into a name.
    reading = CertificateReading(**(GOOD_READING | {"model": "OKTAVIA"}))
    check = check_certificate(reading, MATCHING_REQUEST, 2026)
    assert check.certificate is not None
    assert check.certificate.model == "OKTAVIA"
    assert check.issues == (DocumentIssue.MODEL_MISMATCH,)


def test_cyrillic_letters_in_plate_are_normalised() -> None:
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
def test_unusable_photos_ask_for_new_one(override: dict[str, object], issue: DocumentIssue) -> None:
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
def test_mismatches_go_to_manager(
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
