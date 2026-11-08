"""crm-setup and the adapter against a real EspoCRM. Skipped unless one is given, e.g. the one
docker compose starts:

    LOAN_AGENTS_TEST_ESPOCRM_URL=http://127.0.0.1:8090 \\
    LOAN_AGENTS_TEST_ESPOCRM_PASSWORD=<ESPOCRM_ADMIN_PASSWORD> \\
        pytest tests/test_espocrm_live.py

It sets the CRM up as `loan-agents crm-setup` does and leaves test deals and contacts behind.
"""

from __future__ import annotations

import os
import random
import uuid
from collections.abc import AsyncIterator

import pytest

from loan_agents.adapters.espocrm import (
    API_USER_NAME,
    CASE_FIELD,
    ROLE_NAME,
    EspoClient,
    EspoCrm,
    SetupReport,
    admin_http,
    api_key_http,
    set_up_crm,
)
from loan_agents.contracts import CrmDeal, DealStage
from loan_agents.settings import Settings
from tests.conftest import TODAY

SITE_URL = os.environ.get("LOAN_AGENTS_TEST_ESPOCRM_URL", "")
ADMIN_USER = os.environ.get("LOAN_AGENTS_TEST_ESPOCRM_USER", "admin")
ADMIN_PASSWORD = os.environ.get("LOAN_AGENTS_TEST_ESPOCRM_PASSWORD", "")
pytestmark = pytest.mark.skipif(
    not (SITE_URL and ADMIN_PASSWORD),
    reason="set LOAN_AGENTS_TEST_ESPOCRM_URL and LOAN_AGENTS_TEST_ESPOCRM_PASSWORD",
)


def _random_phone() -> str:
    return "+38067" + "".join(random.choices("0123456789", k=7))


def _deal(phone: str) -> CrmDeal:
    return CrmDeal(
        case_id=f"live-{uuid.uuid4().hex[:12]}",
        title="Кредит під заставу: SKODA OCTAVIA 2018, 200 000 грн",
        first_name="Тарас",
        last_name="Тестовий",
        middle_name="Григорович",
        phone=phone,
        amount_uah=200_000,
        stage=DealStage.REVIEW,
        comment="Тестова заявка",
    )


async def _set_up() -> SetupReport:
    async with admin_http(SITE_URL, ADMIN_USER, ADMIN_PASSWORD) as http:
        return await set_up_crm(EspoClient(http))


@pytest.fixture(scope="module")
def stages() -> dict[DealStage, str]:
    return Settings(_env_file=None).espocrm_stages


@pytest.fixture
async def bot_client() -> AsyncIterator[EspoClient]:
    report = await _set_up()
    async with api_key_http(SITE_URL, report.api_key) as http:
        yield EspoClient(http)


async def test_setup_twice_gives_one_role_one_user_and_the_same_key() -> None:
    first = await _set_up()
    second = await _set_up()
    assert second.api_key == first.api_key
    async with admin_http(SITE_URL, ADMIN_USER, ADMIN_PASSWORD) as http:
        admin = EspoClient(http)
        roles = await admin.get(
            "Role",
            params={
                "where[0][type]": "equals",
                "where[0][attribute]": "name",
                "where[0][value]": ROLE_NAME,
            },
        )
        users = await admin.get(
            "User",
            params={
                "where[0][type]": "equals",
                "where[0][attribute]": "userName",
                "where[0][value]": API_USER_NAME,
            },
        )
        layout = await admin.get("Opportunity/layout/detail")
    assert roles["total"] == 1
    assert users["total"] == 1
    cells = [cell for panel in layout for row in panel["rows"] for cell in row if cell]
    assert [cell["name"] for cell in cells].count(CASE_FIELD) == 1


async def test_a_case_goes_through_the_real_crm(
    bot_client: EspoClient, stages: dict[DealStage, str]
) -> None:
    crm = EspoCrm(bot_client, stages, site_url=SITE_URL, today=lambda: TODAY)
    assert "signed in as" in await crm.check()

    deal = _deal(_random_phone())
    refs = await crm.upsert_deal(deal)
    corrected = deal.model_copy(update={"phone": _random_phone(), "amount_uah": 150_000})
    assert await crm.upsert_deal(corrected) == refs
    await crm.add_comment(refs.deal_id, "Чернетка договору готова")
    await crm.set_stage(refs.deal_id, DealStage.APPROVED)

    stored = await bot_client.get(f"Opportunity/{refs.deal_id}")
    assert stored[CASE_FIELD] == deal.case_id
    assert stored["contactId"] == refs.contact_id
    assert (stored["amount"], stored["amountCurrency"]) == (150_000, "UAH")
    assert stored["stage"] == stages[DealStage.APPROVED]
    contact = await bot_client.get(f"Contact/{refs.contact_id}")
    assert (contact["lastName"], contact["firstName"]) == ("Тестовий", "Тарас")
    assert [entry["phoneNumber"] for entry in contact["phoneNumberData"]] == [corrected.phone]
    stream = await bot_client.get(f"Opportunity/{refs.deal_id}/stream")
    assert "Чернетка договору готова" in [note.get("post") for note in stream["list"]]


async def test_a_namesake_with_another_phone_is_not_refused_as_a_duplicate(
    bot_client: EspoClient, stages: dict[DealStage, str]
) -> None:
    crm = EspoCrm(bot_client, stages, site_url=SITE_URL, today=lambda: TODAY)
    first = await crm.upsert_deal(_deal(_random_phone()))
    second = await crm.upsert_deal(_deal(_random_phone()))
    assert second.contact_id != first.contact_id
