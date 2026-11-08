"""An in-memory EspoCRM for the adapter tests, modelled on answers from a real EspoCRM 10.0.11.

It knows the calls the adapter and crm-setup make, with the same statuses, X-Status-Reason texts
and record shapes. Anything else answers 501, so a test that strays from them fails loudly.
"""

from __future__ import annotations

import base64
import itertools
import json
import re
from collections.abc import Callable
from typing import Any

import httpx

SITE_URL = "http://crm.test"
API_KEY = "test-api-key"
ADMIN_USER = "admin"
ADMIN_PASSWORD = "admin-pass"
STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
NAME_PATTERN = re.compile(r"[^<>=]+")
CUSTOM_FIELD_NAME = re.compile(r"c[A-Z]")

Handler = Callable[[httpx.Request, re.Match[str]], httpx.Response]


def _ok(body: object) -> httpx.Response:
    return httpx.Response(200, json=body)


def _fail(status: int, reason: str = "", body: object = None) -> httpx.Response:
    headers = {"X-Status-Reason": reason} if reason else {}
    if body is None:
        return httpx.Response(status, headers=headers)
    return httpx.Response(status, headers=headers, json=body)


def _invalid(entity: str, field: str, kind: str) -> httpx.Response:
    reason = f"Field validation failure; entityType: {entity}, field: {field}, type: {kind}."
    return _fail(400, reason, {"messageTranslation": {"label": "validationFailure"}})


def _name_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record.get("firstName") or "").casefold(), (record.get("lastName") or "").casefold()


class FakeEspoCrm:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, dict[str, Any]]] = {
            entity: {} for entity in ("Contact", "Opportunity", "Note", "Role", "User")
        }
        self.settings: dict[str, Any] = {
            "currencyList": ["USD"],
            "defaultCurrency": "USD",
            "baseCurrency": "USD",
        }
        self.custom_fields: dict[str, dict[str, Any]] = {}
        self.detail_layout: list[dict[str, Any]] = [
            {
                "rows": [
                    [{"name": "name"}, {"name": "account"}],
                    [{"name": "stage"}, {"name": "amount"}],
                    [{"name": "description"}],
                ]
            }
        ]
        self.requests: list[tuple[str, str]] = []
        # Served instead of an answer, oldest first: a status code, or an exception to raise.
        self.failures: list[int | Exception] = []
        self._ids = itertools.count(1)
        self._routes: list[tuple[str, re.Pattern[str], Handler, bool]] = []
        self._route("GET", r"App/user", self._app_user)
        self._route("GET", r"Metadata", self._metadata)
        self._route("GET", r"Settings", lambda request, match: _ok(self.settings))
        self._route("PUT", r"Settings", self._put_settings, admin=True)
        self._route("POST", r"Admin/fieldManager/Opportunity", self._add_field, admin=True)
        self._route("GET", r"Admin/fieldManager/Opportunity/(\w+)", self._read_field, admin=True)
        self._route("GET", r"Opportunity/layout/detail", lambda r, m: _ok(self.detail_layout))
        self._route("PUT", r"Opportunity/layout/detail", self._put_layout, admin=True)
        self._route("POST", r"Contact", self._create_contact)
        self._route("POST", r"Opportunity", self._create_opportunity)
        self._route("POST", r"Note", self._create_note)
        self._route("POST", r"Role", self._create_role, admin=True)
        self._route("POST", r"User", self._create_user, admin=True)
        self._route("GET", r"(Contact|Opportunity|Role|User)", self._search)
        self._route("GET", r"(Contact|Opportunity)/(\w+)", self._read)
        self._route("PUT", r"(Contact|Opportunity|Role|User)/(\w+)", self._update)
        self._route(
            "DELETE", r"(Contact|Opportunity)/(\w+)", lambda r, m: _fail(403, "No delete access.")
        )

    # --- plumbing ------------------------------------------------------------------------------

    def _route(self, method: str, pattern: str, handler: Handler, *, admin: bool = False) -> None:
        self._routes.append((method, re.compile(pattern), handler, admin))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v1/")
        self.requests.append((request.method, path))
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return _fail(failure, "injected failure")
        signed_in_as = self._signed_in_as(request)
        if signed_in_as is None:
            return _fail(401)
        for method, pattern, handler, admin_only in self._routes:
            match = pattern.fullmatch(path)
            if method == request.method and match:
                if admin_only and signed_in_as != "admin":
                    return _fail(403)
                return handler(request, match)
        return _fail(501, f"the fake does not know {request.method} {path}")

    def _signed_in_as(self, request: httpx.Request) -> str | None:
        if request.headers.get("X-Api-Key") == API_KEY:
            return "api"
        users = [user for user in self.records["User"].values() if user.get("apiKey")]
        if any(request.headers.get("X-Api-Key") == user["apiKey"] for user in users):
            return "api"
        expected = base64.b64encode(f"{ADMIN_USER}:{ADMIN_PASSWORD}".encode()).decode()
        if request.headers.get("Espo-Authorization") == expected:
            return "admin"
        return None

    def _new(self, entity: str, data: dict[str, Any]) -> dict[str, Any]:
        record_id = f"{next(self._ids):017x}"
        record = {"id": record_id, "deleted": False, **data}
        self.records[entity][record_id] = record
        return record

    @staticmethod
    def _body(request: httpx.Request) -> dict[str, Any]:
        return json.loads(request.content)

    # --- reads ---------------------------------------------------------------------------------

    def _app_user(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        return _ok({"user": {"id": "1", "userName": "loan-intake-bot", "type": "api"}})

    def _metadata(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        key = request.url.params.get("key", "")
        if key == "entityDefs.Opportunity.fields.stage.options":
            return _ok(STAGES)
        prefix = "entityDefs.Opportunity.fields."
        if key.startswith(prefix) and key.removeprefix(prefix) in self.custom_fields:
            return _ok(self.custom_fields[key.removeprefix(prefix)])
        return httpx.Response(200)  # a key EspoCRM does not have: empty body

    def _search(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        records = list(self.records[match.group(1)].values())
        params = request.url.params
        if "where[0][attribute]" in params:
            attribute = params["where[0][attribute]"]
            known = {"phoneNumber", "name", "userName", *self.custom_fields}
            if attribute not in known:
                return _fail(400, f"Not existing attribute '{attribute}' in where.")
            value = params["where[0][value]"]
            records = [record for record in records if record.get(attribute) == value]
        size = int(params.get("maxSize", 20))
        return _ok({"total": len(records), "list": records[:size]})

    def _read(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        record = self.records[match.group(1)].get(match.group(2))
        if record is None:
            return _fail(404, f"Record {match.group(2)} does not exist.")
        return _ok(record)

    def _read_field(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        name = match.group(1)
        if name not in self.custom_fields:
            return _fail(500, f"Can't read field defs Opportunity.{name}.")
        return _ok(self.custom_fields[name])

    # --- writes --------------------------------------------------------------------------------

    def _put_settings(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        self.settings |= self._body(request)
        return _ok(self.settings)

    def _add_field(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        defs = self._body(request)
        name = "c" + defs["name"][:1].upper() + defs["name"][1:]
        if name in self.custom_fields:
            return _fail(409, f"Field '{name}' already exists in 'Opportunity'.")
        stored = {key: value for key, value in defs.items() if key != "name"}
        self.custom_fields[name] = stored | {"isCustom": True}
        return _ok(self.custom_fields[name])

    def _put_layout(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        self.detail_layout = json.loads(request.content)
        return _ok(True)

    def _create_contact(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        data = self._body(request)
        if not data.get("lastName"):
            return _invalid("Contact", "lastName", "required")
        if request.headers.get("X-Skip-Duplicate-Check", "").lower() != "true":
            same_name = [
                contact
                for contact in self.records["Contact"].values()
                if _name_key(contact) == _name_key(data)
            ]
            if same_name:
                return _fail(409, "duplicate", same_name)
        return _ok(self._new("Contact", self._with_phone_data(data)))

    @staticmethod
    def _with_phone_data(data: dict[str, Any]) -> dict[str, Any]:
        # Setting phoneNumber replaces the primary number; the old one does not stay.
        if "phoneNumber" in data:
            data["phoneNumberData"] = [
                {"phoneNumber": data["phoneNumber"], "type": "Mobile", "primary": True}
            ]
        return data

    def _opportunity_problem(
        self, data: dict[str, Any], *, creating: bool
    ) -> httpx.Response | None:
        if creating:
            for field in ("name", "amount", "closeDate"):
                if data.get(field) in (None, ""):
                    return _invalid("Opportunity", field, "required")
        if "name" in data and not NAME_PATTERN.fullmatch(data["name"]):
            return _invalid("Opportunity", "name", "pattern")
        if "amountCurrency" in data and data["amountCurrency"] not in self.settings["currencyList"]:
            return _invalid("Opportunity", "amount", "validCurrency")
        if "stage" in data and data["stage"] not in STAGES:
            return _invalid("Opportunity", "stage", "valid")
        return None

    def _create_opportunity(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        data = self._body(request)
        problem = self._opportunity_problem(data, creating=True)
        if problem is not None:
            return problem
        # A custom field EspoCRM does not have is dropped without a word.
        data = {
            key: value
            for key, value in data.items()
            if not CUSTOM_FIELD_NAME.match(key) or key in self.custom_fields
        }
        data.setdefault("stage", "Prospecting")
        if data.get("contactId"):
            data["contactsIds"] = [data["contactId"]]
        return _ok(self._new("Opportunity", data))

    def _create_note(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        data = self._body(request)
        if data.get("parentId") not in self.records.get(data.get("parentType", ""), {}):
            return _fail(403, "No create access.")
        return _ok(self._new("Note", data))

    def _create_role(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        return _ok(self._new("Role", self._body(request)))

    def _create_user(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        data = self._body(request)
        if any(user["userName"] == data["userName"] for user in self.records["User"].values()):
            return _fail(409, "userNameExists")
        if data.get("authMethod") == "ApiKey":
            data["apiKey"] = f"key-{next(self._ids)}"
        return _ok(self._new("User", data))

    def _update(self, request: httpx.Request, match: re.Match[str]) -> httpx.Response:
        entity, record_id = match.group(1), match.group(2)
        record = self.records[entity].get(record_id)
        if record is None:
            return _fail(404, f"Record {record_id} not found.")
        data = self._body(request)
        if entity == "Opportunity":
            problem = self._opportunity_problem(data, creating=False)
            if problem is not None:
                return problem
            # The case field is read-only after create: a change is ignored, not refused.
            data = {key: value for key, value in data.items() if key not in self.custom_fields}
            if data.get("contactId"):
                data["contactsIds"] = [data["contactId"]]
        if entity == "Contact":
            data = self._with_phone_data(data)
        record.update(data)
        return _ok(record)

    # --- what tests look at ----------------------------------------------------------------------

    def contact(self, contact_id: str) -> dict[str, Any]:
        return self.records["Contact"][contact_id]

    def deal(self, deal_id: str) -> dict[str, Any]:
        return self.records["Opportunity"][deal_id]

    def posts(self, deal_id: str) -> list[str]:
        return [
            note["post"]
            for note in self.records["Note"].values()
            if note.get("parentType") == "Opportunity" and note.get("parentId") == deal_id
        ]

    def set_up(self) -> FakeEspoCrm:
        """The state `loan-agents crm-setup` leaves behind, for tests that skip running it."""
        self.settings["currencyList"] = ["USD", "UAH"]
        self.custom_fields["cLoanCaseId"] = {"type": "varchar", "readOnlyAfterCreate": True}
        return self
