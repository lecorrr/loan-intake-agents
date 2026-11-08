"""The real Bitrix24 adapter against the bundled mock, plus retry behaviour on fake transports."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from loan_agents.adapters.bitrix24 import ORIGINATOR_ID, Bitrix24Client, Bitrix24Crm
from loan_agents.contracts import CrmDeal, DealStage
from loan_agents.mock_bitrix24 import create_mock_app
from loan_agents.ports import CrmError

SECRET = "s3cret"
STAGES = {
    DealStage.NEW: "NEW",
    DealStage.REVIEW: "PREPARATION",
    DealStage.APPROVED: "EXECUTING",
    DealStage.REJECTED: "LOSE",
}
DEAL = CrmDeal(
    case_id="tg-42",
    title="Кредит під заставу: SKODA OCTAVIA 2018",
    first_name="Тарас",
    last_name="Шевченко",
    middle_name="Григорович",
    phone="+380671234567",
    amount_uah=200_000,
    stage=DealStage.REVIEW,
    comment="summary",
)


async def _no_sleep(seconds: float) -> None:
    return None


def _client(transport: httpx.AsyncBaseTransport, **kwargs: object) -> Bitrix24Client:
    http = httpx.AsyncClient(transport=transport, base_url="http://b24")
    return Bitrix24Client(
        f"http://b24/rest/1/{SECRET}/", http, min_interval_s=0, sleep=_no_sleep, **kwargs
    )


@pytest.fixture
def mock_app():
    return create_mock_app(SECRET)


@pytest.fixture
def crm(mock_app) -> Bitrix24Crm:
    return Bitrix24Crm(_client(httpx.ASGITransport(app=mock_app)), STAGES)


async def test_upsert_creates_a_contact_and_a_linked_deal(crm: Bitrix24Crm, mock_app) -> None:
    refs = await crm.upsert_deal(DEAL)
    store = mock_app.state.store
    deal = store.deals[int(refs.deal_id)]
    assert deal["ORIGINATOR_ID"] == ORIGINATOR_ID
    assert deal["ORIGIN_ID"] == "tg-42"
    assert deal["STAGE_ID"] == "PREPARATION"
    assert deal["CURRENCY_ID"] == "UAH"
    assert deal["CONTACT_ID"] == refs.contact_id
    assert _phones(store.contacts[int(refs.contact_id)]) == ["+380671234567"]


def _phones(contact: dict) -> list[str]:
    return [entry["VALUE"] for entry in contact["PHONE"]]


async def test_a_second_upsert_updates_instead_of_duplicating(crm: Bitrix24Crm, mock_app) -> None:
    first = await crm.upsert_deal(DEAL)
    second = await crm.upsert_deal(DEAL.model_copy(update={"amount_uah": 150_000}))
    store = mock_app.state.store
    assert second == first
    assert len(store.deals) == 1
    assert len(store.contacts) == 1
    assert store.deals[int(first.deal_id)]["OPPORTUNITY"] == 150_000
    assert _phones(store.contacts[int(first.contact_id)]) == ["+380671234567"]


async def test_corrected_details_replace_the_old_ones(crm: Bitrix24Crm, mock_app) -> None:
    refs = await crm.upsert_deal(DEAL)
    corrected = DEAL.model_copy(update={"phone": "+380501112233", "last_name": "Шевчук"})
    await crm.upsert_deal(corrected)
    await crm.upsert_deal(corrected)
    contact = mock_app.state.store.contacts[int(refs.contact_id)]
    assert _phones(contact) == ["+380501112233"]
    assert contact["LAST_NAME"] == "Шевчук"


async def test_comments_and_stages_reach_the_deal(crm: Bitrix24Crm, mock_app) -> None:
    refs = await crm.upsert_deal(DEAL)
    await crm.add_comment(refs.deal_id, "Чернетка договору готова")
    await crm.set_stage(refs.deal_id, DealStage.APPROVED)
    store = mock_app.state.store
    assert store.comments[int(refs.deal_id)] == ["Чернетка договору готова"]
    assert store.deals[int(refs.deal_id)]["STAGE_ID"] == "EXECUTING"


async def test_a_wrong_webhook_code_fails_without_retrying(mock_app) -> None:
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=mock_app), base_url="http://b24")
    client = Bitrix24Client("http://b24/rest/1/wrong/", http, min_interval_s=0, sleep=_no_sleep)
    with pytest.raises(CrmError, match="INVALID_CREDENTIALS"):
        await client.call("crm.deal.list", {})


def _scripted(responses: list[Callable[[], httpx.Response]]) -> httpx.MockTransport:
    calls = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(calls)()

    return httpx.MockTransport(handler)


def _json(status: int, body: object) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(status, content=json.dumps(body).encode())


def _network_error() -> httpx.Response:
    raise httpx.ConnectError("connection refused")


async def test_rate_limit_and_network_errors_are_retried() -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    transport = _scripted(
        [
            _json(503, {"error": "QUERY_LIMIT_EXCEEDED", "error_description": "Too many"}),
            _network_error,
            _json(200, {"result": 7, "time": {}}),
        ]
    )
    http = httpx.AsyncClient(transport=transport, base_url="http://b24")
    client = Bitrix24Client(
        f"http://b24/rest/1/{SECRET}/", http, min_interval_s=0, sleep=record_sleep
    )
    assert await client.call("crm.deal.add", {"fields": {"TITLE": "x"}}) == 7
    assert sleeps == [1.0, 2.0]


async def test_retries_stop_at_the_limit() -> None:
    transport = _scripted([_json(503, {"error": "QUERY_LIMIT_EXCEEDED"})] * 2)
    with pytest.raises(CrmError, match="gave up after 2 attempts"):
        await _client(transport, max_attempts=2).call("crm.deal.list", {})


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (lambda: httpx.Response(200, content=b"<html>"), "not JSON"),
        (_json(200, {"time": {}}), "no result"),
        (_json(400, {"error": "INVALID_ARG_VALUE", "error_description": "bad"}), "INVALID_ARG"),
    ],
)
async def test_unusable_answers_are_errors(response, message: str) -> None:
    with pytest.raises(CrmError, match=message):
        await _client(_scripted([response])).call("crm.deal.list", {})


def test_every_stage_needs_a_bitrix24_id() -> None:
    with pytest.raises(ValueError, match="no Bitrix24 stage id"):
        Bitrix24Crm(_client(_scripted([])), {DealStage.NEW: "NEW"})


async def test_mock_rejects_unknown_methods(mock_app) -> None:
    with pytest.raises(CrmError, match="ERROR_METHOD_NOT_FOUND"):
        await _client(httpx.ASGITransport(app=mock_app)).call("crm.lead.add", {})
