"""The EspoCRM adapter and crm-setup against tests/fake_espocrm.py, built from EspoCRM 10."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import httpx
import pytest
from pydantic import SecretStr

from loan_agents.adapters import espocrm
from loan_agents.adapters.espocrm import (
    API_USER_NAME,
    CASE_FIELD,
    EspoClient,
    EspoCrm,
    SetupReport,
    admin_http,
    api_key_http,
    set_up_crm,
)
from loan_agents.cli import _check_espocrm, main, save_api_key
from loan_agents.contracts import Channel, CrmDeal, DealStage, DecisionAction, ManagerDecision
from loan_agents.fakes import RecordingMessenger
from loan_agents.ports import CrmError
from loan_agents.settings import ConfigError, Settings
from loan_agents.wiring import build_runtime
from tests.conftest import TODAY, sample_bytes
from tests.fake_espocrm import ADMIN_PASSWORD, ADMIN_USER, API_KEY, SITE_URL, FakeEspoCrm

STAGES = {
    DealStage.NEW: "Prospecting",
    DealStage.REVIEW: "Qualification",
    DealStage.APPROVED: "Negotiation",
    DealStage.REJECTED: "Closed Lost",
}
DECISION_DAY = date(2026, 11, 9)
DEAL = CrmDeal(
    case_id="tg-42",
    title="Кредит під заставу: SKODA OCTAVIA 2018, 200 000 грн",
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


class Clock:
    def __init__(self) -> None:
        self.day = TODAY

    def __call__(self) -> date:
        return self.day


@pytest.fixture
def fake() -> FakeEspoCrm:
    return FakeEspoCrm().set_up()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
async def crm(fake: FakeEspoCrm, clock: Clock) -> AsyncIterator[EspoCrm]:
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        yield EspoCrm(EspoClient(http, sleep=_no_sleep), STAGES, site_url=SITE_URL, today=clock)


# --- deals, contacts, comments ---------------------------------------------------------------


async def test_upsert_creates_a_contact_and_a_linked_deal(crm: EspoCrm, fake: FakeEspoCrm) -> None:
    refs = await crm.upsert_deal(DEAL)
    deal = fake.deal(refs.deal_id)
    assert deal[CASE_FIELD] == "tg-42"
    assert deal["contactId"] == refs.contact_id
    assert (deal["amount"], deal["amountCurrency"]) == (200_000, "UAH")
    assert (deal["stage"], deal["closeDate"]) == ("Qualification", TODAY.isoformat())
    assert deal["description"] == "summary"
    contact = fake.contact(refs.contact_id)
    assert (contact["lastName"], contact["firstName"], contact["middleName"]) == (
        "Шевченко",
        "Тарас",
        "Григорович",
    )
    assert contact["phoneNumber"] == "+380671234567"
    assert refs.deal_url == f"{SITE_URL}/#Opportunity/view/{refs.deal_id}"


async def test_a_second_upsert_updates_instead_of_duplicating(
    crm: EspoCrm, fake: FakeEspoCrm
) -> None:
    first = await crm.upsert_deal(DEAL)
    second = await crm.upsert_deal(DEAL.model_copy(update={"amount_uah": 150_000}))
    assert second == first
    assert len(fake.records["Opportunity"]) == 1
    assert len(fake.records["Contact"]) == 1
    assert fake.deal(first.deal_id)["amount"] == 150_000


async def test_corrected_details_replace_the_old_ones(crm: EspoCrm, fake: FakeEspoCrm) -> None:
    refs = await crm.upsert_deal(DEAL)
    await crm.upsert_deal(DEAL.model_copy(update={"phone": "+380501112233", "last_name": "Шевчук"}))
    contact = fake.contact(refs.contact_id)
    assert contact["lastName"] == "Шевчук"
    assert [entry["phoneNumber"] for entry in contact["phoneNumberData"]] == ["+380501112233"]


async def test_a_returning_client_keeps_one_contact(crm: EspoCrm, fake: FakeEspoCrm) -> None:
    first = await crm.upsert_deal(DEAL)
    second = await crm.upsert_deal(
        DEAL.model_copy(update={"case_id": "tg-43", "first_name": "ТАРАС"})
    )
    assert second.contact_id == first.contact_id
    assert second.deal_id != first.deal_id


@pytest.mark.parametrize(
    "change",
    [
        # EspoCRM's own check matches on the name alone and would refuse this one with a 409.
        {"phone": "+380501112233"},
        {"first_name": "Олена", "last_name": "Коваленко", "middle_name": ""},
    ],
    ids=["namesake-with-another-phone", "another-person-on-the-same-phone"],
)
async def test_people_who_share_only_a_name_or_a_phone_get_contacts_of_their_own(
    crm: EspoCrm, fake: FakeEspoCrm, change: dict[str, str]
) -> None:
    first = await crm.upsert_deal(DEAL)
    second = await crm.upsert_deal(DEAL.model_copy(update={"case_id": "tg-43", **change}))
    assert second.contact_id != first.contact_id
    assert len(fake.records["Contact"]) == 2


async def test_a_name_that_could_not_be_split_goes_into_the_required_last_name(
    crm: EspoCrm, fake: FakeEspoCrm
) -> None:
    whole = DEAL.model_copy(
        update={"first_name": "Тарас Шевченко", "last_name": "", "middle_name": ""}
    )
    refs = await crm.upsert_deal(whole)
    contact = fake.contact(refs.contact_id)
    assert (contact["firstName"], contact["lastName"]) == ("", "Тарас Шевченко")


async def test_characters_espocrm_refuses_in_a_deal_name_are_dropped(
    crm: EspoCrm, fake: FakeEspoCrm
) -> None:
    refs = await crm.upsert_deal(DEAL.model_copy(update={"title": "Кредит: <BMW> X5=2019"}))
    assert fake.deal(refs.deal_id)["name"] == "Кредит: BMW X5 2019"


async def test_comments_and_stage_changes_reach_the_deal(
    crm: EspoCrm, fake: FakeEspoCrm, clock: Clock
) -> None:
    refs = await crm.upsert_deal(DEAL)
    await crm.add_comment(refs.deal_id, "Чернетка договору готова")
    clock.day = DECISION_DAY
    await crm.set_stage(refs.deal_id, DealStage.APPROVED)
    assert fake.posts(refs.deal_id) == ["Чернетка договору готова"]
    deal = fake.deal(refs.deal_id)
    assert (deal["stage"], deal["closeDate"]) == ("Negotiation", DECISION_DAY.isoformat())


async def test_a_whole_case_lands_in_espocrm(
    fake: FakeEspoCrm, crm: EspoCrm, settings: Settings
) -> None:
    messenger = RecordingMessenger()
    async with build_runtime(
        settings, fake=True, crm=crm, messenger=messenger, today=lambda: TODAY
    ) as runtime:
        service = runtime.service
        await service.start("web-1", Channel.WEB)
        for text in ("Мене звати Тарас Шевченко", "067 123 45 67", "200 тис грн на 12 місяців"):
            await service.handle_text("web-1", Channel.WEB, text)
        await service.handle_text("web-1", Channel.WEB, "Шкода Октавія 2018 року")
        await service.handle_document(
            "web-1", Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )
        await service.decide(
            "web-1", ManagerDecision(action=DecisionAction.APPROVE, manager="test")
        )
    (deal_id,) = fake.records["Opportunity"]
    assert f"CRM: {SITE_URL}/#Opportunity/view/{deal_id}" in messenger.approval_requests[0].card
    assert fake.deal(deal_id)["stage"] == "Negotiation"
    posts = fake.posts(deal_id)
    assert len(posts) == 2  # the contract draft, then the decision
    assert "test" in posts[1]


# --- how requests fail -------------------------------------------------------------------------


async def test_a_wrong_api_key_fails_at_once(fake: FakeEspoCrm) -> None:
    async with api_key_http(SITE_URL, "wrong-key", transport=fake.transport()) as http:
        crm = EspoCrm(EspoClient(http, sleep=_no_sleep), STAGES, site_url=SITE_URL)
        with pytest.raises(CrmError, match="401: the API key or the password was not accepted"):
            await crm.upsert_deal(DEAL)
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    ("method", "failure", "attempts"),
    [
        ("GET", 503, 2),
        ("GET", httpx.ReadTimeout("slow"), 2),
        ("PUT", 502, 2),
        # A POST that never left can be sent again; one that may have arrived can't.
        ("POST", httpx.ConnectError("refused"), 2),
        ("POST", httpx.ReadTimeout("slow"), 1),
        ("POST", 503, 1),
        ("GET", 500, 1),
        ("GET", 404, 1),
    ],
)
async def test_only_requests_that_cannot_apply_twice_are_retried(
    fake: FakeEspoCrm, method: str, failure: int | Exception, attempts: int
) -> None:
    fake.records["Contact"]["c1"] = {"id": "c1", "lastName": "Шевченко"}
    requests = {
        "GET": ("Contact", None),
        "PUT": ("Contact/c1", {"firstName": "Тарас"}),
        "POST": ("Contact", {"lastName": "Коваленко"}),
    }
    path, body = requests[method]
    fake.failures.append(failure)
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        client = EspoClient(http, sleep=_no_sleep)
        if attempts == 1:
            with pytest.raises(CrmError):
                await client.request(method, path, body=body)
        else:
            await client.request(method, path, body=body)
    assert len(fake.requests) == attempts


async def test_retries_stop_at_the_limit_with_growing_pauses(fake: FakeEspoCrm) -> None:
    sleeps: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    fake.failures.extend([503, 503, 503])
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        with pytest.raises(CrmError, match="HTTP 503"):
            await EspoClient(http, sleep=record_sleep).get("Contact")
    assert len(fake.requests) == 3
    assert sleeps == [0.5, 1.0]


async def test_errors_keep_the_records_espocrm_sends_back_out(fake: FakeEspoCrm) -> None:
    fake.records["Contact"]["c1"] = {
        "id": "c1",
        "firstName": "Тарас",
        "lastName": "Шевченко",
        "phoneNumber": "+380671234567",
    }
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        with pytest.raises(CrmError) as caught:
            await EspoClient(http).post("Contact", {"firstName": "Тарас", "lastName": "Шевченко"})
    message = str(caught.value)
    assert "409" in message and "duplicate" in message
    assert "Шевченко" not in message and "380671234567" not in message


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, content=b"<html>"), "not JSON"),
        (httpx.Response(200, json={"total": 0}), "expected a list"),
        (
            httpx.Response(301, headers={"Location": "https://crm.test/api/v1/X?phone=380"}),
            "redirected to https://crm.test/api/v1/X; check ESPOCRM_URL",
        ),
    ],
)
async def test_unusable_answers_are_errors(response: httpx.Response, message: str) -> None:
    transport = httpx.MockTransport(lambda request: response)
    async with api_key_http(SITE_URL, API_KEY, transport=transport) as http:
        crm = EspoCrm(EspoClient(http), STAGES, site_url=SITE_URL)
        with pytest.raises(CrmError, match=message):
            await crm.upsert_deal(DEAL)


def test_every_stage_needs_a_name() -> None:
    client = EspoClient(httpx.AsyncClient())
    with pytest.raises(ValueError, match="no EspoCRM stage"):
        EspoCrm(client, {DealStage.NEW: "Prospecting"}, site_url=SITE_URL)


# --- doctor -------------------------------------------------------------------------------------


async def test_check_passes_on_a_crm_that_is_set_up(crm: EspoCrm) -> None:
    assert "signed in as loan-intake-bot" in await crm.check()


@pytest.mark.parametrize(
    ("break_it", "message"),
    [
        (lambda fake: fake.custom_fields.clear(), f"no {CASE_FIELD} field"),
        (lambda fake: fake.settings.update(currencyList=["USD"]), "UAH is not an allowed"),
    ],
    ids=["case-field", "currency"],
)
async def test_check_names_what_crm_setup_would_fix(
    crm: EspoCrm, fake: FakeEspoCrm, break_it, message: str
) -> None:
    break_it(fake)
    with pytest.raises(CrmError, match=message):
        await crm.check()


async def test_check_names_stages_espocrm_does_not_have(fake: FakeEspoCrm) -> None:
    stages = STAGES | {DealStage.APPROVED: "Approved"}
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        crm = EspoCrm(EspoClient(http), stages, site_url=SITE_URL)
        with pytest.raises(CrmError, match=r"stages EspoCRM does not have: \['Approved'\]"):
            await crm.check()


@pytest.fixture
def through_fake(monkeypatch: pytest.MonkeyPatch, fake: FakeEspoCrm) -> FakeEspoCrm:
    """Every EspoCRM connection the CLI opens goes to the fake."""
    for name in ("api_key_http", "admin_http"):
        opener = getattr(espocrm, name)

        def patched(*args, _opener=opener, **kwargs):
            return _opener(*args, transport=fake.transport(), **kwargs)

        monkeypatch.setattr(espocrm, name, patched)
    return fake


async def test_doctor_checks_the_crm_with_the_api_key(
    through_fake: FakeEspoCrm, settings: Settings
) -> None:
    configured = settings.model_copy(
        update={"espocrm_url": SITE_URL, "espocrm_api_key": SecretStr(API_KEY)}
    )
    assert (await _check_espocrm(configured)).startswith(f"{SITE_URL}: signed in as")


async def test_doctor_asks_for_the_key_when_only_the_url_is_set(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="run: loan-agents crm-setup"):
        await _check_espocrm(settings.model_copy(update={"espocrm_url": SITE_URL}))


# --- crm-setup ----------------------------------------------------------------------------------


async def _set_up(fake: FakeEspoCrm) -> SetupReport:
    async with admin_http(SITE_URL, ADMIN_USER, ADMIN_PASSWORD, transport=fake.transport()) as http:
        return await set_up_crm(EspoClient(http, sleep=_no_sleep))


async def test_setup_prepares_a_fresh_crm_for_the_app() -> None:
    fake = FakeEspoCrm()
    report = await _set_up(fake)
    assert fake.settings == {
        "currencyList": ["USD", "UAH"],
        "defaultCurrency": "UAH",
        "baseCurrency": "UAH",
    }
    assert fake.custom_fields[CASE_FIELD]["readOnlyAfterCreate"] is True
    assert [{"name": CASE_FIELD}, False] in fake.detail_layout[0]["rows"]
    (role,) = fake.records["Role"].values()
    assert role["data"]["Opportunity"]["stream"] == "all"
    assert role["data"]["Contact"]["delete"] == "no"
    (user,) = fake.records["User"].values()
    assert (user["userName"], user["type"], user["rolesIds"]) == (
        API_USER_NAME,
        "api",
        [role["id"]],
    )

    async with api_key_http(SITE_URL, report.api_key, transport=fake.transport()) as http:
        crm = EspoCrm(EspoClient(http), STAGES, site_url=SITE_URL)
        await crm.check()
        refs = await crm.upsert_deal(DEAL)
    assert fake.deal(refs.deal_id)[CASE_FIELD] == DEAL.case_id


async def test_running_setup_again_gives_the_same_key_and_no_duplicates() -> None:
    fake = FakeEspoCrm()
    first = await _set_up(fake)
    second = await _set_up(fake)
    assert second.api_key == first.api_key
    assert len(fake.records["Role"]) == 1
    assert len(fake.records["User"]) == 1
    case_rows = [row for row in fake.detail_layout[0]["rows"] if row[0] == {"name": CASE_FIELD}]
    assert len(case_rows) == 1


async def test_setup_leaves_the_main_currency_of_a_crm_in_use_alone() -> None:
    fake = FakeEspoCrm()
    fake.records["Opportunity"]["o1"] = {"id": "o1", "name": "an older deal"}
    await _set_up(fake)
    assert fake.settings["baseCurrency"] == "USD"
    assert fake.settings["currencyList"] == ["USD", "UAH"]


async def test_setup_puts_back_a_disabled_api_user() -> None:
    fake = FakeEspoCrm()
    first = await _set_up(fake)
    (user,) = fake.records["User"].values()
    user.update(isActive=False, rolesIds=[])
    second = await _set_up(fake)
    assert user["isActive"] is True
    assert len(user["rolesIds"]) == 1
    assert second.api_key == first.api_key


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        (
            lambda fake: fake.records["User"].update(
                u1={"id": "u1", "userName": API_USER_NAME, "type": "regular"}
            ),
            "not an API user",
        ),
        (lambda fake: fake.custom_fields.update({CASE_FIELD: {"type": "int"}}), "exists as a int"),
    ],
    ids=["regular-user-with-the-bot-name", "case-field-of-another-type"],
)
async def test_setup_refuses_to_take_over_what_it_did_not_make(prepare, message: str) -> None:
    fake = FakeEspoCrm()
    prepare(fake)
    with pytest.raises(CrmError, match=message):
        await _set_up(fake)


async def test_setup_needs_an_administrator() -> None:
    fake = FakeEspoCrm()
    async with api_key_http(SITE_URL, API_KEY, transport=fake.transport()) as http:
        with pytest.raises(CrmError, match="HTTP 403"):
            await set_up_crm(EspoClient(http))


@pytest.fixture
def admin_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> Settings:
    configured = settings.model_copy(
        update={
            "espocrm_url": SITE_URL,
            "espocrm_admin_username": ADMIN_USER,
            "espocrm_admin_password": SecretStr(ADMIN_PASSWORD),
        }
    )
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: configured)
    return configured


def test_crm_setup_command_prints_the_line_for_env(
    through_fake: FakeEspoCrm, admin_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["crm-setup"]) == 0
    (user,) = through_fake.records["User"].values()
    assert f"ESPOCRM_API_KEY={user['apiKey']}" in capsys.readouterr().out


def test_crm_setup_can_save_the_key_in_env(
    through_fake: FakeEspoCrm,
    admin_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ESPOCRM_URL=http://crm.test\nESPOCRM_API_KEY=\nLOG_LEVEL=INFO\n")
    monkeypatch.setattr("loan_agents.cli.ENV_FILE", env_file)
    assert main(["crm-setup", "--write-env"]) == 0
    (user,) = through_fake.records["User"].values()
    assert env_file.read_text() == (
        f"ESPOCRM_URL=http://crm.test\nESPOCRM_API_KEY={user['apiKey']}\nLOG_LEVEL=INFO\n"
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (
            "A=1\nESPOCRM_API_KEY=old\nESPOCRM_API_KEY=older\nB=2\n",
            "A=1\nESPOCRM_API_KEY=new\nB=2\n",
        ),
        ("A=1\n", "A=1\nESPOCRM_API_KEY=new\n"),
    ],
    ids=["replaces-and-drops-repeats", "adds-a-missing-line"],
)
def test_the_key_ends_up_on_one_line_of_env(tmp_path, before: str, after: str) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(before)
    save_api_key(env_file, "new")
    assert env_file.read_text() == after


def test_saving_the_key_needs_an_env_file(tmp_path) -> None:
    with pytest.raises(ConfigError, match="by hand"):
        save_api_key(tmp_path / ".env", "new")


def test_crm_setup_says_when_espocrm_is_not_answering(
    through_fake: FakeEspoCrm,
    admin_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    through_fake.failures.extend([httpx.ConnectError("refused")] * 3)
    assert main(["crm-setup"]) == 1
    assert "may still be starting" in capsys.readouterr().err


def test_crm_setup_explains_a_wrong_password(
    through_fake: FakeEspoCrm,
    admin_settings: Settings,
    capsys: pytest.CaptureFixture[str],
) -> None:
    admin_settings.espocrm_admin_password = SecretStr("wrong")
    assert main(["crm-setup"]) == 1
    error = capsys.readouterr().err
    assert "401" in error
    assert "first start" in error


def test_serve_stops_before_starting_when_the_crm_key_is_missing(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    def server_started(*args: object, **kwargs: object) -> None:
        raise AssertionError("the server started")

    monkeypatch.setattr("uvicorn.run", server_started)
    half_done = settings.model_copy(update={"espocrm_url": SITE_URL})
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: half_done)
    assert main(["serve"]) == 2
    assert "crm-setup --write-env" in capsys.readouterr().err
