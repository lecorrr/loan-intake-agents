"""A Bitrix24-compatible inbound-webhook server for demos: the same URLs and JSON as a real portal.

Bitrix24's free plan has no REST webhooks, so this stands in until a paid plan or trial is used.
Point B24_WEBHOOK_URL at http://127.0.0.1:8090/rest/1/<MOCK_B24_SECRET>/ and open
http://127.0.0.1:8090/ to watch deals arrive. Data lives in memory and is gone on restart.

Only the methods this project calls are implemented, plus crm.deal.fields for `loan-agents doctor`.
"""

from __future__ import annotations

import html
import secrets
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse


class _BadArgument(ValueError):
    pass


class _Store:
    def __init__(self) -> None:
        self.contacts: dict[int, dict[str, Any]] = {}
        self.deals: dict[int, dict[str, Any]] = {}
        self.comments: dict[int, list[str]] = {}
        self._next_id = 0

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id


def _fields(params: dict[str, Any]) -> dict[str, Any]:
    fields = params.get("fields")
    if not isinstance(fields, dict):
        raise _BadArgument("fields must be an object")
    return fields


def _entity(table: dict[int, dict[str, Any]], kind: str, params: dict[str, Any]) -> dict[str, Any]:
    raw_id = params.get("id", params.get("ID"))
    try:
        entity_id = int(raw_id)
    except (TypeError, ValueError) as error:
        raise _BadArgument("id must be an integer") from error
    if entity_id not in table:
        raise _BadArgument(f"{kind} {entity_id} not found")
    return table[entity_id]


def _deal(store: _Store, params: dict[str, Any]) -> dict[str, Any]:
    return _entity(store.deals, "deal", params)


def _merge_phones(
    store: _Store, current: list[dict[str, Any]], changes: Any
) -> list[dict[str, Any]]:
    """Bitrix24 multifield rules: an entry with an ID changes or deletes that entry, one without
    an ID is added. Sending a list never replaces the whole set."""
    if not isinstance(changes, list) or not all(isinstance(item, dict) for item in changes):
        raise _BadArgument("PHONE must be a list of objects")
    phones = [dict(entry) for entry in current]
    for change in changes:
        if "ID" not in change:
            phones.append(
                {
                    "ID": str(store.new_id()),
                    "TYPE_ID": "PHONE",
                    "VALUE": str(change.get("VALUE") or ""),
                    "VALUE_TYPE": change.get("VALUE_TYPE", "WORK"),
                }
            )
            continue
        match = next((entry for entry in phones if entry["ID"] == str(change["ID"])), None)
        if match is None:
            raise _BadArgument(f"PHONE entry {change['ID']} not found")
        if change.get("DELETE") == "Y" or not change.get("VALUE"):
            phones.remove(match)
        else:
            match["VALUE"] = str(change["VALUE"])
            match["VALUE_TYPE"] = change.get("VALUE_TYPE", match["VALUE_TYPE"])
    return phones


def _contact_add(store: _Store, params: dict[str, Any]) -> int:
    fields = dict(_fields(params))
    phones = _merge_phones(store, [], fields.pop("PHONE", []))
    contact_id = store.new_id()
    store.contacts[contact_id] = {"ID": str(contact_id), **fields, "PHONE": phones}
    return contact_id


def _contact_get(store: _Store, params: dict[str, Any]) -> dict[str, Any]:
    return _entity(store.contacts, "contact", params)


def _contact_update(store: _Store, params: dict[str, Any]) -> bool:
    contact = _contact_get(store, params)
    fields = dict(_fields(params))
    if "PHONE" in fields:
        contact["PHONE"] = _merge_phones(store, contact["PHONE"], fields.pop("PHONE"))
    contact.update(fields)
    return True


def _deal_add(store: _Store, params: dict[str, Any]) -> int:
    fields = _fields(params)
    if not str(fields.get("TITLE") or "").strip():
        raise _BadArgument("TITLE is required")
    deal_id = store.new_id()
    store.deals[deal_id] = {
        "ID": str(deal_id),
        "STAGE_ID": "NEW",
        **{key: value for key, value in fields.items()},
        "CONTACT_ID": str(fields.get("CONTACT_ID") or ""),
    }
    store.comments[deal_id] = []
    return deal_id


def _deal_update(store: _Store, params: dict[str, Any]) -> bool:
    _deal(store, params).update(_fields(params))
    return True


def _deal_get(store: _Store, params: dict[str, Any]) -> dict[str, Any]:
    return _deal(store, params)


def _deal_list(store: _Store, params: dict[str, Any]) -> list[dict[str, Any]]:
    wanted = params.get("filter") or {}
    select = params.get("select") or []
    if not isinstance(wanted, dict) or not isinstance(select, list):
        raise _BadArgument("filter must be an object and select a list")
    found = [
        deal
        for deal in store.deals.values()
        if all(str(deal.get(key, "")) == str(value) for key, value in wanted.items())
    ]
    if select:
        return [{key: deal.get(key) for key in select} for deal in found]
    return found


def _timeline_comment_add(store: _Store, params: dict[str, Any]) -> int:
    fields = _fields(params)
    if fields.get("ENTITY_TYPE") != "deal":
        raise _BadArgument("this mock supports ENTITY_TYPE=deal only")
    deal = _deal(store, {"id": fields.get("ENTITY_ID")})
    comment = str(fields.get("COMMENT") or "")
    if not comment.strip():
        raise _BadArgument("COMMENT is required")
    store.comments[int(deal["ID"])].append(comment)
    return store.new_id()


def _deal_fields(store: _Store, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "ID": {"type": "integer", "isReadOnly": True},
        "TITLE": {"type": "string", "isRequired": True},
        "STAGE_ID": {"type": "crm_status"},
        "OPPORTUNITY": {"type": "double"},
        "ORIGINATOR_ID": {"type": "string"},
        "ORIGIN_ID": {"type": "string"},
    }


_METHODS: dict[str, Callable[[_Store, dict[str, Any]], Any]] = {
    "crm.contact.add": _contact_add,
    "crm.contact.get": _contact_get,
    "crm.contact.update": _contact_update,
    "crm.deal.add": _deal_add,
    "crm.deal.update": _deal_update,
    "crm.deal.get": _deal_get,
    "crm.deal.list": _deal_list,
    "crm.deal.fields": _deal_fields,
    "crm.timeline.comment.add": _timeline_comment_add,
}


def _error(status: int, code: str, description: str) -> JSONResponse:
    return JSONResponse({"error": code, "error_description": description}, status_code=status)


def create_mock_app(secret: str) -> FastAPI:
    if not secret:
        raise ValueError("the mock needs a non-empty webhook secret")
    app = FastAPI(title="Mock Bitrix24")
    store = _Store()
    app.state.store = store

    @app.post("/rest/{user_id}/{code}/{method}")
    async def call(user_id: int, code: str, method: str, request: Request) -> JSONResponse:
        started = time.time()
        if not secrets.compare_digest(code, secret):
            return _error(401, "INVALID_CREDENTIALS", "Invalid request credentials")
        handler = _METHODS.get(method.removesuffix(".json"))
        if handler is None:
            return _error(404, "ERROR_METHOD_NOT_FOUND", "Method not found!")
        try:
            params = await request.json()
        except ValueError:
            return _error(400, "INVALID_REQUEST", "The body must be JSON")
        if not isinstance(params, dict):
            return _error(400, "INVALID_REQUEST", "The body must be a JSON object")
        try:
            result = handler(store, params)
        except _BadArgument as error:
            return _error(400, "INVALID_ARG_VALUE", str(error))
        finished = time.time()
        timing = {"start": started, "finish": finished, "duration": finished - started}
        return JSONResponse({"result": result, "time": timing})

    @app.get("/", response_class=HTMLResponse)
    async def page() -> str:
        rows = []
        for deal_id in sorted(store.deals, reverse=True):
            deal = store.deals[deal_id]
            contact = store.contacts.get(int(deal["CONTACT_ID"] or 0), {})
            phones = ", ".join(phone.get("VALUE", "") for phone in contact.get("PHONE", []))
            name = " ".join(
                str(contact.get(key, "")) for key in ("LAST_NAME", "NAME", "SECOND_NAME")
            )
            comments = "".join(
                f"<pre>{html.escape(comment)}</pre>" for comment in store.comments[deal_id]
            )
            rows.append(
                "<tr>"
                f"<td>{deal_id}</td><td>{html.escape(str(deal.get('TITLE', '')))}</td>"
                f"<td>{html.escape(str(deal.get('OPPORTUNITY', '')))} UAH</td>"
                f"<td>{html.escape(str(deal.get('STAGE_ID', '')))}</td>"
                f"<td>{html.escape(name.strip())}<br>{html.escape(phones)}</td>"
                f"<td><pre>{html.escape(str(deal.get('COMMENTS', '')))}</pre>{comments}</td>"
                "</tr>"
            )
        body = "".join(rows) or "<tr><td colspan=6>No deals yet</td></tr>"
        return (
            "<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=5>"
            "<title>Mock Bitrix24</title>"
            "<style>body{font-family:sans-serif;margin:24px}td,th{border:1px solid #ccc;"
            "padding:6px;vertical-align:top}table{border-collapse:collapse}"
            "pre{white-space:pre-wrap;margin:0 0 8px}</style>"
            "<h1>Mock Bitrix24: deals</h1><p>Refreshes every 5 s. In-memory only.</p>"
            "<table><tr><th>ID</th><th>Title</th><th>Amount</th><th>Stage</th><th>Contact</th>"
            f"<th>Comments</th></tr>{body}</table>"
        )

    return app
