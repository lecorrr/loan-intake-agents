"""HTTP API: the web channel, manager decisions and health; runs Telegram polling if set up.

Web case ids are random, so the id grants access; Telegram cases and decisions need the
manager token.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from loan_agents import __version__
from loan_agents.contracts import CaseView, Channel, DecisionAction, ManagerDecision, TurnResult
from loan_agents.ports import Messenger
from loan_agents.service import (
    CaseNotFound,
    DecisionNotPending,
    InputError,
    LoanIntakeService,
    TurnFailed,
)
from loan_agents.settings import Settings, load_settings
from loan_agents.telegram_bot import build_dispatcher, polling
from loan_agents.wiring import Runtime, build_runtime

log = logging.getLogger(__name__)

_WEB_PREFIX = "web-"


class TextIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)


class DecisionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    comment: str | None = Field(None, max_length=1000)
    manager: str = Field("api", min_length=1, max_length=100)


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def _service(request: Request) -> LoanIntakeService:
    return _runtime(request).service


def _require_manager(request: Request, token: str | None) -> None:
    expected = _runtime(request).settings.manager_api_token
    if expected is None or not expected.get_secret_value():
        raise HTTPException(403, "manager API disabled: set MANAGER_API_TOKEN")
    if token is None or not secrets.compare_digest(token, expected.get_secret_value()):
        raise HTTPException(403, "invalid manager token")


def _require_web_case(case_id: str) -> None:
    if not case_id.startswith(_WEB_PREFIX):
        raise HTTPException(403, "only web cases can be written through this endpoint")


def _report_polling_end(app: FastAPI, task: asyncio.Task[None]) -> None:
    if task.cancelled():
        app.state.telegram_status = "stopped"
        return
    error = task.exception()
    if error is not None:
        log.error("step=telegram event=polling_died error=%r", error)
        app.state.telegram_status = f"failed: {type(error).__name__}: {error}"
    else:
        app.state.telegram_status = "stopped"


def create_app(
    settings: Settings | None = None,
    *,
    fake: bool = False,
    messenger: Messenger | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = settings or load_settings()
        async with build_runtime(active, fake=fake, messenger=messenger) as runtime:
            app.state.runtime = runtime
            app.state.telegram_status = "disabled"
            async with contextlib.AsyncExitStack() as stack:
                if runtime.bot is not None and active.manager_chat_id is not None:
                    dispatcher = build_dispatcher(
                        runtime.service,
                        runtime.chat_cases,
                        runtime.bot,
                        active.manager_chat_id,
                        active.max_upload_bytes,
                    )
                    await stack.enter_async_context(
                        polling(
                            dispatcher, runtime.bot, lambda task: _report_polling_end(app, task)
                        )
                    )
                    app.state.telegram_status = "polling"
                log.info(
                    "loan-intake-agents %s started fake=%s settings=%s",
                    __version__,
                    fake,
                    active.describe(),
                )
                yield

    app = FastAPI(title="Loan intake agents", version=__version__, lifespan=lifespan)

    @app.exception_handler(InputError)
    async def _input_error(request: Request, error: InputError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.exception_handler(CaseNotFound)
    async def _not_found(request: Request, error: CaseNotFound) -> JSONResponse:
        return JSONResponse({"detail": f"no case {error}"}, status_code=404)

    @app.exception_handler(DecisionNotPending)
    async def _not_pending(request: Request, error: DecisionNotPending) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(TurnFailed)
    async def _turn_failed(request: Request, error: TurnFailed) -> JSONResponse:
        return JSONResponse({"detail": "the agents failed on this turn; see the log"}, 503)

    @app.get("/health")
    async def health(request: Request) -> dict[str, str]:
        return {
            "status": "ok",
            "version": __version__,
            "telegram": request.app.state.telegram_status,
        }

    @app.post("/api/cases", status_code=201)
    async def create_case(request: Request) -> TurnResult:
        case_id = f"{_WEB_PREFIX}{uuid.uuid4().hex[:16]}"
        return await _service(request).start(case_id, Channel.WEB)

    @app.post("/api/cases/{case_id}/messages")
    async def post_message(case_id: str, body: TextIn, request: Request) -> TurnResult:
        _require_web_case(case_id)
        return await _service(request).handle_text(case_id, Channel.WEB, body.text)

    @app.post("/api/cases/{case_id}/documents")
    async def post_document(
        case_id: str,
        request: Request,
        file: Annotated[UploadFile, File()],
    ) -> TurnResult:
        _require_web_case(case_id)
        limit = _runtime(request).settings.max_upload_bytes
        data = await file.read(limit + 1)
        mime_type = file.content_type or ""
        return await _service(request).handle_document(case_id, Channel.WEB, data, mime_type)

    @app.get("/api/cases/{case_id}")
    async def get_case(
        case_id: str,
        request: Request,
        x_manager_token: Annotated[str | None, Header()] = None,
    ) -> CaseView:
        if not case_id.startswith(_WEB_PREFIX):
            _require_manager(request, x_manager_token)
        return await _service(request).view(case_id)

    @app.post("/api/cases/{case_id}/decision")
    async def post_decision(
        case_id: str,
        body: DecisionIn,
        request: Request,
        x_manager_token: Annotated[str | None, Header()] = None,
    ) -> TurnResult:
        _require_manager(request, x_manager_token)
        decision = ManagerDecision(action=body.action, manager=body.manager, comment=body.comment)
        return await _service(request).decide(case_id, decision)

    return app
