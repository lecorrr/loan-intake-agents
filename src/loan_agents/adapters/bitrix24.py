"""Bitrix24 through an inbound webhook: POST {webhook}{method}.json with a JSON body.

Quirks handled here and nowhere else:
- answers are {"result": ...}; failures are {"error": CODE, "error_description": ...};
- the portal's request bucket drains at 2 requests/s on most plans and answers 503
  QUERY_LIMIT_EXCEEDED when full, so calls are spaced and those errors are retried;
- ORIGINATOR_ID / ORIGIN_ID link a deal to our case, which makes upsert a find-or-create and keeps
  a retried step from creating a duplicate deal.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from loan_agents.contracts import CrmDeal, CrmRefs, DealStage
from loan_agents.ports import CrmError

log = logging.getLogger(__name__)

ORIGINATOR_ID = "loan-intake-agents"
_RETRYABLE_ERRORS = frozenset({"QUERY_LIMIT_EXCEEDED", "OPERATION_TIME_LIMIT"})


class Bitrix24Client:
    def __init__(
        self,
        webhook_url: str,
        http: httpx.AsyncClient,
        *,
        min_interval_s: float = 0.55,
        max_attempts: int = 4,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._base = webhook_url.rstrip("/") + "/"
        self._http = http
        self._min_interval_s = min_interval_s
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def _space_calls(self) -> None:
        wait_s = self._last_call + self._min_interval_s - time.monotonic()
        if wait_s > 0:
            await self._sleep(wait_s)
        self._last_call = time.monotonic()

    async def call(self, method: str, params: dict[str, Any]) -> Any:
        url = f"{self._base}{method}.json"
        async with self._lock:
            for attempt in range(1, self._max_attempts + 1):
                await self._space_calls()
                outcome = await self._attempt(url, method, params)
                if outcome.retry_reason is None:
                    return outcome.result
                if attempt == self._max_attempts:
                    raise CrmError(
                        f"bitrix24 {method}: gave up after {attempt} attempts: "
                        f"{outcome.retry_reason}"
                    )
                backoff_s = min(8.0, 0.5 * 2**attempt)
                log.warning(
                    "step=bitrix24 method=%s attempt=%d retry_in_s=%.1f reason=%s",
                    method,
                    attempt,
                    backoff_s,
                    outcome.retry_reason,
                )
                await self._sleep(backoff_s)
        raise AssertionError("unreachable: the loop returns or raises")

    async def _attempt(self, url: str, method: str, params: dict[str, Any]) -> _Outcome:
        try:
            response = await self._http.post(url, json=params)
        except httpx.TransportError as error:
            return _Outcome(retry_reason=f"network: {type(error).__name__}: {error}")
        try:
            body = response.json()
        except ValueError as error:
            if response.status_code >= 500:
                return _Outcome(retry_reason=f"HTTP {response.status_code}, body is not JSON")
            raise CrmError(
                f"bitrix24 {method}: HTTP {response.status_code}, body is not JSON"
            ) from error
        if isinstance(body, dict) and "error" in body:
            code = str(body["error"])
            if code in _RETRYABLE_ERRORS or response.status_code >= 500:
                return _Outcome(retry_reason=code)
            raise CrmError(f"bitrix24 {method}: {code}: {body.get('error_description', '')}")
        if response.status_code != 200 or not isinstance(body, dict) or "result" not in body:
            raise CrmError(f"bitrix24 {method}: HTTP {response.status_code}, no result in answer")
        return _Outcome(result=body["result"])


@dataclass(frozen=True)
class _Outcome:
    result: Any = None
    retry_reason: str | None = None


class Bitrix24Crm:
    def __init__(self, client: Bitrix24Client, stage_ids: dict[DealStage, str]) -> None:
        missing = [stage for stage in DealStage if not stage_ids.get(stage)]
        if missing:
            raise ValueError(f"no Bitrix24 stage id for {[stage.value for stage in missing]}")
        self._client = client
        self._stage_ids = stage_ids

    async def _find_deal(self, case_id: str) -> dict[str, Any] | None:
        found = await self._client.call(
            "crm.deal.list",
            {
                "filter": {"ORIGINATOR_ID": ORIGINATOR_ID, "ORIGIN_ID": case_id},
                "select": ["ID", "CONTACT_ID"],
            },
        )
        if not isinstance(found, list):
            raise CrmError(f"bitrix24 crm.deal.list: expected a list, got {type(found).__name__}")
        if len(found) > 1:
            log.warning("step=bitrix24 case=%s event=duplicate_deals count=%d", case_id, len(found))
        return found[0] if found else None

    def _deal_fields(self, deal: CrmDeal) -> dict[str, Any]:
        return {
            "TITLE": deal.title,
            "OPPORTUNITY": deal.amount_uah,
            "CURRENCY_ID": "UAH",
            "STAGE_ID": self._stage_ids[deal.stage],
            "COMMENTS": deal.comment,
            "ORIGINATOR_ID": ORIGINATOR_ID,
            "ORIGIN_ID": deal.case_id,
        }

    @staticmethod
    def _name_fields(deal: CrmDeal) -> dict[str, Any]:
        return {
            "NAME": deal.first_name,
            "LAST_NAME": deal.last_name,
            "SECOND_NAME": deal.middle_name,
        }

    async def _add_contact(self, deal: CrmDeal) -> str:
        fields = self._name_fields(deal) | {
            "PHONE": [{"VALUE": deal.phone, "VALUE_TYPE": "MOBILE"}],
            "ORIGINATOR_ID": ORIGINATOR_ID,
            "ORIGIN_ID": deal.case_id,
        }
        return str(await self._client.call("crm.contact.add", {"fields": fields}))

    async def _update_contact(self, contact_id: str, deal: CrmDeal) -> None:
        current = await self._client.call("crm.contact.get", {"id": contact_id})
        if not isinstance(current, dict):
            raise CrmError(f"bitrix24 crm.contact.get {contact_id}: expected an object")
        fields = self._name_fields(deal)
        phones = current.get("PHONE") or []
        if deal.phone not in {entry.get("VALUE") for entry in phones}:
            # An entry sent with its ID replaces that entry; one without an ID is added next to
            # the old ones, so a corrected number would leave the wrong one on the contact.
            entry = {"VALUE": deal.phone, "VALUE_TYPE": "MOBILE"}
            if phones and phones[0].get("ID"):
                entry["ID"] = phones[0]["ID"]
            fields["PHONE"] = [entry]
        await self._client.call("crm.contact.update", {"id": contact_id, "fields": fields})

    async def upsert_deal(self, deal: CrmDeal) -> CrmRefs:
        existing = await self._find_deal(deal.case_id)
        if existing is None:
            contact_id = await self._add_contact(deal)
            deal_id = await self._client.call(
                "crm.deal.add",
                {"fields": self._deal_fields(deal) | {"CONTACT_ID": contact_id}},
            )
            return CrmRefs(contact_id=contact_id, deal_id=str(deal_id))

        deal_id = str(existing["ID"])
        fields = self._deal_fields(deal)
        contact_id = str(existing.get("CONTACT_ID") or "")
        if contact_id in ("", "0"):
            contact_id = await self._add_contact(deal)
            fields["CONTACT_ID"] = contact_id
        else:
            await self._update_contact(contact_id, deal)
        await self._client.call("crm.deal.update", {"id": deal_id, "fields": fields})
        return CrmRefs(contact_id=contact_id, deal_id=deal_id)

    async def add_comment(self, deal_id: str, text: str) -> None:
        await self._client.call(
            "crm.timeline.comment.add",
            {"fields": {"ENTITY_ID": int(deal_id), "ENTITY_TYPE": "deal", "COMMENT": text}},
        )

    async def set_stage(self, deal_id: str, stage: DealStage) -> None:
        await self._client.call(
            "crm.deal.update", {"id": deal_id, "fields": {"STAGE_ID": self._stage_ids[stage]}}
        )
