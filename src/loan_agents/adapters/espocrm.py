"""EspoCRM REST client ({site}/api/v1/{Entity}[/{id}]), signed in as an API user.

Quirks seen on EspoCRM 10.0:
- errors are a status plus X-Status-Reason; bodies can hold other records and are never shown;
- Contact.lastName and Opportunity.closeDate are required; Opportunity.name can't contain < > =;
- creating a contact runs a name-only duplicate check, so contacts are matched on phone and name;
- setting phoneNumber replaces the primary number;
- the case id lives in custom field cLoanCaseId, read-only after create; upsert is find-or-create.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from loan_agents.contracts import CrmDeal, CrmRefs, DealStage
from loan_agents.ports import CrmError

log = logging.getLogger(__name__)

CASE_FIELD = "cLoanCaseId"
CURRENCY = "UAH"
API_USER_NAME = "loan-intake-bot"
ROLE_NAME = "Loan intake bot"

# GET and PUT are safe to repeat; a POST is retried only if it never left this machine.
_REPLAYABLE_METHODS = frozenset({"GET", "PUT"})
_NEVER_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
_TRANSIENT_STATUSES = frozenset({502, 503, 504})
_NAME_FORBIDDEN_CHARS = re.compile(r"[<>=]")


def api_url(site_url: str) -> str:
    return site_url.rstrip("/") + "/api/v1/"


def deal_url(site_url: str, deal_id: str) -> str:
    return f"{site_url.rstrip('/')}/#Opportunity/view/{deal_id}"


def api_key_http(
    site_url: str,
    api_key: str,
    *,
    timeout_s: float = 15,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        base_url=api_url(site_url),
        headers={"X-Api-Key": api_key},
        timeout=timeout_s,
        transport=transport,
    )


def admin_http(
    site_url: str,
    user_name: str,
    password: str,
    *,
    timeout_s: float = 60,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    # EspoCRM's own auth header; some PHP setups drop the standard Authorization header.
    token = base64.b64encode(f"{user_name}:{password}".encode()).decode("ascii")
    return httpx.AsyncClient(
        base_url=api_url(site_url),
        headers={"Espo-Authorization": token},
        timeout=timeout_s,
        transport=transport,
    )


class EspoError(CrmError):
    def __init__(self, method: str, path: str, status: int | None, reason: str) -> None:
        if status is None:
            text = f"no answer: {reason}"
        elif status == 401:
            text = "HTTP 401: the API key or the password was not accepted"
        else:
            text = f"HTTP {status} {reason}".rstrip()
        super().__init__(f"espocrm {method} {path}: {text}")
        self.status = status


class EspoClient:
    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        max_attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._http = http
        self._max_attempts = max_attempts
        self._sleep = sleep

    async def get(self, path: str, params: Mapping[str, str | int] | None = None) -> Any:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: object, headers: Mapping[str, str] | None = None) -> Any:
        return await self.request("POST", path, body=body, headers=headers)

    async def put(self, path: str, body: object) -> Any:
        return await self.request("PUT", path, body=body)

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int] | None = None,
        body: object = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        for attempt in range(1, self._max_attempts + 1):
            last_attempt = attempt == self._max_attempts
            try:
                response = await self._http.request(
                    method, path, params=params, json=body, headers=headers
                )
            except httpx.TransportError as error:
                resendable = method in _REPLAYABLE_METHODS or isinstance(error, _NEVER_SENT_ERRORS)
                if last_attempt or not resendable:
                    reason = f"{type(error).__name__}: {error}"
                    raise EspoError(method, path, None, reason) from error
                retry_reason = type(error).__name__
            else:
                if response.status_code < 300:
                    return _read_body(response, method, path)
                if response.status_code < 400:
                    # Usually http -> https. Query dropped: it can hold a phone number.
                    location = response.headers.get("Location", "?").split("?", 1)[0]
                    reason = f"redirected to {location}; check ESPOCRM_URL"
                    raise EspoError(method, path, response.status_code, reason)
                transient = response.status_code in _TRANSIENT_STATUSES
                if last_attempt or not transient or method not in _REPLAYABLE_METHODS:
                    reason = response.headers.get("X-Status-Reason", "")
                    raise EspoError(method, path, response.status_code, reason)
                retry_reason = f"HTTP {response.status_code}"
            backoff_s = min(4.0, 0.5 * 2 ** (attempt - 1))
            log.warning(
                "step=espocrm method=%s path=%s attempt=%d retry_in_s=%.1f reason=%s",
                method,
                path,
                attempt,
                backoff_s,
                retry_reason,
            )
            await self._sleep(backoff_s)
        raise AssertionError("unreachable: the loop returns or raises")


def _read_body(response: httpx.Response, method: str, path: str) -> Any:
    # Some reads return an empty body, e.g. a metadata key that doesn't exist.
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError as error:
        raise EspoError(method, path, response.status_code, "answer is not JSON") from error


async def _find(
    client: EspoClient, entity: str, attribute: str, value: str, select: str
) -> list[dict[str, Any]]:
    """Records whose attribute equals value, oldest first."""
    found = await client.get(
        entity,
        params={
            "where[0][type]": "equals",
            "where[0][attribute]": attribute,
            "where[0][value]": value,
            "select": select,
            "orderBy": "createdAt",
            "order": "asc",
            "maxSize": 50,
        },
    )
    records = found.get("list") if isinstance(found, dict) else None
    if not isinstance(records, list):
        raise CrmError(f"espocrm GET {entity}: expected a list of records")
    return records


def _record_id(record: object, entity: str) -> str:
    if not isinstance(record, dict) or not isinstance(record.get("id"), str):
        raise CrmError(f"espocrm {entity}: the answer has no record id")
    return record["id"]


def _name_key(record: Mapping[str, Any]) -> tuple[str, str]:
    first = record.get("firstName") or ""
    last = record.get("lastName") or ""
    return first.strip().casefold(), last.strip().casefold()


class EspoCrm:
    def __init__(
        self,
        client: EspoClient,
        stages: Mapping[DealStage, str],
        *,
        site_url: str,
        today: Callable[[], date] = date.today,
    ) -> None:
        missing = [stage.value for stage in DealStage if not stages.get(stage)]
        if missing:
            raise ValueError(f"no EspoCRM stage for {missing}")
        self._client = client
        self._stages = dict(stages)
        self._site_url = site_url
        self._today = today

    def _deal_fields(self, deal: CrmDeal) -> dict[str, Any]:
        return {
            "name": " ".join(_NAME_FORBIDDEN_CHARS.sub(" ", deal.title).split()),
            "amount": deal.amount_uah,
            "amountCurrency": CURRENCY,
            "stage": self._stages[deal.stage],
            # Required; set_stage() later moves it to the day of the decision.
            "closeDate": self._today().isoformat(),
            "description": deal.comment,
        }

    @staticmethod
    def _contact_fields(deal: CrmDeal) -> dict[str, str]:
        # lastName is required, so a name the intake could not split goes there whole.
        if deal.last_name:
            first_name, last_name = deal.first_name, deal.last_name
        else:
            first_name, last_name = "", deal.first_name
        return {
            "firstName": first_name,
            "lastName": last_name,
            "middleName": deal.middle_name,
            "phoneNumber": deal.phone,
        }

    async def _find_contact(self, fields: Mapping[str, str]) -> str | None:
        """A returning client: same phone and same name. Anything less gets a new contact."""
        candidates = await _find(
            self._client, "Contact", "phoneNumber", fields["phoneNumber"], "id,firstName,lastName"
        )
        for candidate in candidates:
            if _name_key(candidate) == _name_key(fields):
                return _record_id(candidate, "Contact")
        return None

    async def _contact_for(self, deal: CrmDeal) -> str:
        fields = self._contact_fields(deal)
        existing = await self._find_contact(fields)
        if existing is not None:
            return existing
        # EspoCRM's duplicate check is by name only; _find_contact has already ruled out a real one.
        created = await self._client.post(
            "Contact", fields, headers={"X-Skip-Duplicate-Check": "true"}
        )
        return _record_id(created, "Contact")

    async def _find_deal(self, case_id: str) -> dict[str, Any] | None:
        found = await _find(self._client, "Opportunity", CASE_FIELD, case_id, "id,contactId")
        if len(found) > 1:
            log.warning("step=espocrm case=%s event=duplicate_deals count=%d", case_id, len(found))
        return found[0] if found else None

    async def upsert_deal(self, deal: CrmDeal) -> CrmRefs:
        existing = await self._find_deal(deal.case_id)
        fields = self._deal_fields(deal)
        if existing is None:
            contact_id = await self._contact_for(deal)
            created = await self._client.post(
                "Opportunity", fields | {"contactId": contact_id, CASE_FIELD: deal.case_id}
            )
            deal_id = _record_id(created, "Opportunity")
        else:
            deal_id = _record_id(existing, "Opportunity")
            contact_id = existing.get("contactId") or ""
            if contact_id:
                await self._client.put(f"Contact/{contact_id}", self._contact_fields(deal))
            else:
                contact_id = await self._contact_for(deal)
                fields["contactId"] = contact_id
            await self._client.put(f"Opportunity/{deal_id}", fields)
        return CrmRefs(
            contact_id=contact_id, deal_id=deal_id, deal_url=deal_url(self._site_url, deal_id)
        )

    async def add_comment(self, deal_id: str, text: str) -> None:
        await self._client.post(
            "Note", {"type": "Post", "parentType": "Opportunity", "parentId": deal_id, "post": text}
        )

    async def set_stage(self, deal_id: str, stage: DealStage) -> None:
        await self._client.put(
            f"Opportunity/{deal_id}",
            {"stage": self._stages[stage], "closeDate": self._today().isoformat()},
        )

    async def check(self) -> str:
        """Signs in and checks what the deals need; raises CrmError naming every problem."""
        me = await self._client.get("App/user")
        user_name = ((me or {}).get("user") or {}).get("userName")
        options = await self._client.get(
            "Metadata", params={"key": "entityDefs.Opportunity.fields.stage.options"}
        )
        case_field = await self._client.get(
            "Metadata", params={"key": f"entityDefs.Opportunity.fields.{CASE_FIELD}"}
        )
        settings = await self._client.get("Settings")
        currencies = (settings or {}).get("currencyList") or []

        problems = []
        unknown = sorted(set(self._stages.values()) - set(options or []))
        if unknown:
            problems.append(f"ESPOCRM_STAGE_* name stages EspoCRM does not have: {unknown}")
        if not case_field:
            problems.append(f"Opportunity has no {CASE_FIELD} field (run: loan-agents crm-setup)")
        if CURRENCY not in currencies:
            problems.append(f"{CURRENCY} is not an allowed currency (run: loan-agents crm-setup)")
        if problems:
            raise CrmError("; ".join(problems))
        return f"signed in as {user_name}; stages, {CASE_FIELD} and {CURRENCY} are in place"


_ROLE_ACCESS = {"create": "yes", "read": "all", "edit": "all", "delete": "no", "stream": "all"}
# "stream" lets the bot comment on deals.
_ROLE_DATA = {"Contact": _ROLE_ACCESS, "Opportunity": _ROLE_ACCESS}
_CASE_FIELD_DEFS = {
    "name": "loanCaseId",  # EspoCRM saves it as cLoanCaseId
    "type": "varchar",
    "maxLength": 64,
    "readOnlyAfterCreate": True,
    "label": "Номер заявки",
}


@dataclass(frozen=True)
class SetupReport:
    api_key: str
    steps: list[str]


async def set_up_crm(admin: EspoClient) -> SetupReport:
    """Creates or repairs what the app needs. Safe to run again; it never deletes anything."""
    steps = [
        await _allow_currency(admin),
        await _add_case_field(admin),
        await _show_case_field(admin),
    ]
    role_id, role_step = await _ensure_role(admin)
    api_key, user_step = await _ensure_api_user(admin, role_id)
    return SetupReport(api_key=api_key, steps=[*steps, role_step, user_step])


async def _allow_currency(admin: EspoClient) -> str:
    settings = await admin.get("Settings") or {}
    currencies = list(settings.get("currencyList") or [])
    update: dict[str, Any] = {}
    if CURRENCY not in currencies:
        update["currencyList"] = [*currencies, CURRENCY]
    if settings.get("baseCurrency") != CURRENCY and await _deal_count(admin) == 0:
        # Only an empty CRM gets UAH as base currency; one in use keeps its own.
        update["baseCurrency"] = CURRENCY
        update["defaultCurrency"] = CURRENCY
    if not update:
        return f"currency: {CURRENCY} already allowed"
    await admin.put("Settings", update)
    return f"currency: set {', '.join(sorted(update))} for {CURRENCY}"


async def _deal_count(admin: EspoClient) -> int | None:
    found = await admin.get("Opportunity", params={"maxSize": 1, "select": "id"})
    total = found.get("total") if isinstance(found, dict) else None
    return total if isinstance(total, int) else None


async def _add_case_field(admin: EspoClient) -> str:
    try:
        await admin.post("Admin/fieldManager/Opportunity", _CASE_FIELD_DEFS)
        return f"field: Opportunity.{CASE_FIELD} created"
    except EspoError as error:
        if error.status != 409:  # 409: the field exists already
            raise
    existing = await admin.get(f"Admin/fieldManager/Opportunity/{CASE_FIELD}") or {}
    if existing.get("type") != "varchar":
        raise CrmError(
            f"Opportunity.{CASE_FIELD} exists as a {existing.get('type')} field; "
            "the app needs a varchar. Remove it in Administration > Entity Manager and run again."
        )
    return f"field: Opportunity.{CASE_FIELD} already there"


async def _show_case_field(admin: EspoClient) -> str:
    layout = await admin.get("Opportunity/layout/detail")
    if not (isinstance(layout, list) and layout and isinstance(layout[0], dict)):
        return "deal card: unexpected layout, add the case field by hand if you want to see it"
    shown = {
        cell.get("name")
        for panel in layout
        if isinstance(panel, dict)
        for row in panel.get("rows") or []
        for cell in row
        if isinstance(cell, dict)
    }
    if CASE_FIELD in shown:
        return "deal card: shows the case number already"
    rows = layout[0].setdefault("rows", [])
    rows.insert(1, [{"name": CASE_FIELD}, False])
    await admin.put("Opportunity/layout/detail", layout)
    return "deal card: case number added"


async def _ensure_role(admin: EspoClient) -> tuple[str, str]:
    found = await _find(admin, "Role", "name", ROLE_NAME, "id")
    if found:
        role_id = _record_id(found[0], "Role")
        # The app owns this role, so its access is reset to what the app needs.
        await admin.put(f"Role/{role_id}", {"data": _ROLE_DATA})
        return role_id, f"role: '{ROLE_NAME}' already there, access checked"
    created = await admin.post("Role", {"name": ROLE_NAME, "data": _ROLE_DATA})
    return _record_id(created, "Role"), f"role: '{ROLE_NAME}' created"


async def _ensure_api_user(admin: EspoClient, role_id: str) -> tuple[str, str]:
    found = await _find(
        admin, "User", "userName", API_USER_NAME, "id,type,authMethod,apiKey,isActive,rolesIds"
    )
    if not found:
        created = await admin.post(
            "User",
            {
                "userName": API_USER_NAME,
                "type": "api",
                "authMethod": "ApiKey",
                "rolesIds": [role_id],
                "isActive": True,
            },
        )
        return _api_key(created), f"API user: '{API_USER_NAME}' created"
    user = found[0]
    if user.get("type") != "api" or user.get("authMethod") != "ApiKey":
        raise CrmError(
            f"a user named '{API_USER_NAME}' exists but is not an API user with an API key; "
            "rename or remove it in EspoCRM and run again"
        )
    roles = list(user.get("rolesIds") or [])
    if role_id in roles and user.get("isActive"):
        return _api_key(user), f"API user: '{API_USER_NAME}' already there"
    if role_id not in roles:
        roles.append(role_id)
    await admin.put(f"User/{_record_id(user, 'User')}", {"rolesIds": roles, "isActive": True})
    return _api_key(user), f"API user: '{API_USER_NAME}' given its role and activated"


def _api_key(user: object) -> str:
    key = user.get("apiKey") if isinstance(user, dict) else None
    if not isinstance(key, str) or not key:
        raise CrmError(f"espocrm User: no API key for '{API_USER_NAME}'")
    return key
