"""The HTTP API with its real lifespan, on fakes."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from loan_agents.api import create_app
from loan_agents.core import texts_uk as texts
from loan_agents.fakes import RecordingMessenger
from loan_agents.settings import Settings
from tests.conftest import MANAGER_TOKEN, sample_bytes

ANSWERS = (
    "Мене звати Тарас Шевченко",
    "067 123 45 67",
    "200 тис грн на 12 місяців",
    "Шкода Октавія 2018 року",
)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings, fake=True, messenger=RecordingMessenger())
    with TestClient(app) as test_client:
        yield test_client


def _submitted_case(client: TestClient) -> str:
    created = client.post("/api/cases")
    assert created.status_code == 201
    case_id = created.json()["case_id"]
    for text in ANSWERS:
        assert client.post(f"/api/cases/{case_id}/messages", json={"text": text}).status_code == 200
    upload = client.post(
        f"/api/cases/{case_id}/documents",
        files={"file": ("certificate.png", sample_bytes("certificate-ok.png"), "image/png")},
    )
    assert upload.status_code == 200
    assert upload.json()["awaiting_manager"] is True
    return case_id


def test_health_reports_version_and_telegram(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["telegram"] == "disabled"


def test_a_new_case_starts_with_the_greeting(client: TestClient) -> None:
    body = client.post("/api/cases").json()
    assert body["case_id"].startswith("web-")
    assert body["replies"][0] == texts.GREETING


def test_full_application_and_decision(client: TestClient) -> None:
    case_id = _submitted_case(client)
    view = client.get(f"/api/cases/{case_id}").json()
    assert view["awaiting_manager"] is True
    assert view["request"]["phone"] == "+38067***4567"
    assert view["certificate"]["vin"] == "TMB**********3456"

    decided = client.post(
        f"/api/cases/{case_id}/decision",
        json={"action": "approve"},
        headers={"X-Manager-Token": MANAGER_TOKEN},
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "approved"

    again = client.post(
        f"/api/cases/{case_id}/decision",
        json={"action": "approve"},
        headers={"X-Manager-Token": MANAGER_TOKEN},
    )
    assert again.status_code == 409


@pytest.mark.parametrize("headers", [{}, {"X-Manager-Token": "wrong"}])
def test_decisions_need_the_manager_token(client: TestClient, headers: dict[str, str]) -> None:
    case_id = _submitted_case(client)
    response = client.post(
        f"/api/cases/{case_id}/decision", json={"action": "approve"}, headers=headers
    )
    assert response.status_code == 403


def test_telegram_cases_are_closed_to_web_writes_and_anonymous_reads(client: TestClient) -> None:
    assert client.post("/api/cases/tg-42/messages", json={"text": "hi"}).status_code == 403
    assert client.get("/api/cases/tg-42").status_code == 403


def test_unknown_case_is_404_for_the_manager(client: TestClient) -> None:
    response = client.get("/api/cases/tg-42", headers={"X-Manager-Token": MANAGER_TOKEN})
    assert response.status_code == 404


def test_a_non_image_upload_is_422(client: TestClient) -> None:
    case_id = client.post("/api/cases").json()["case_id"]
    response = client.post(
        f"/api/cases/{case_id}/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == texts.UPLOAD_NOT_IMAGE


def test_extra_fields_in_a_message_are_rejected(client: TestClient) -> None:
    case_id = client.post("/api/cases").json()["case_id"]
    response = client.post(f"/api/cases/{case_id}/messages", json={"text": "hi", "admin": True})
    assert response.status_code == 422
