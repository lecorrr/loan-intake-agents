from __future__ import annotations

from dataclasses import replace

import pytest

from loan_agents.contracts import CaseStatus, LoanRequest, VehicleCertificate
from loan_agents.core.masking import (
    mask_name,
    mask_phone,
    mask_plate,
    mask_vin,
    masked_certificate,
    masked_request,
)
from loan_agents.core.routing import RouteFacts, Step, TurnInput, next_step

READY = RouteFacts(
    status=CaseStatus.COLLECTING,
    turn_input=TurnInput.TEXT,
    has_unread_text=False,
    has_unread_document=False,
    missing_fields=(),
    has_certificate=True,
    has_valuation=True,
    has_crm_deal=True,
    needs_contract=False,
)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"status": CaseStatus.APPROVED}, Step.CLOSED),
        ({"status": CaseStatus.REJECTED, "has_unread_text": True}, Step.CLOSED),
        ({"status": CaseStatus.AWAITING_MANAGER}, Step.WAIT),
        ({"has_unread_text": True, "has_unread_document": True}, Step.INTAKE),
        ({"has_unread_document": True}, Step.DOCUMENTS),
        ({"missing_fields": ("phone",), "has_certificate": False}, Step.ASK),
        ({"has_certificate": False, "turn_input": TurnInput.TEXT}, Step.ASK_DOCUMENT),
        ({"has_certificate": False, "turn_input": TurnInput.START}, Step.ASK_DOCUMENT),
        ({"has_certificate": False, "turn_input": TurnInput.DOCUMENT}, Step.WAIT),
        ({"has_valuation": False, "has_crm_deal": False}, Step.VALUATION),
        ({"has_crm_deal": False}, Step.CRM),
        ({"needs_contract": True}, Step.CONTRACT),
        ({}, Step.SUBMIT),
    ],
)
def test_next_step(changes: dict[str, object], expected: Step) -> None:
    assert next_step(replace(READY, **changes)) is expected


def test_masks_keep_only_what_identifies_a_case_to_staff() -> None:
    assert mask_phone("+380671234567") == "+38067***4567"
    assert mask_vin("TMBJJ7NE8J0123456") == "TMB**********3456"
    assert mask_plate("AA1234BB") == "AA****BB"
    assert mask_name("Тарас Шевченко") == "Т. Ш."


def test_masked_views_hide_raw_values() -> None:
    request = LoanRequest(full_name="Тарас Шевченко", phone="+380671234567", amount_uah=200_000)
    assert "+380671234567" not in str(masked_request(request))
    certificate = VehicleCertificate(
        plate_number="AA1234BB",
        vin="TMBJJ7NE8J0123456",
        make="SKODA",
        model="OCTAVIA",
        year=2018,
        owner_full_name="ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ",
    )
    shown = str(masked_certificate(certificate))
    assert "TMBJJ7NE8J0123456" not in shown
    assert "ШЕВЧЕНКО" not in shown
