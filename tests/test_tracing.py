"""Tracing: what core.masking lets out, and what the Langfuse adapter sends for a whole case."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import os
import uuid
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from loan_agents.adapters.tracing_langfuse import LangfuseTracer, check_keys, open_langfuse
from loan_agents.cli import _check_langfuse, main
from loan_agents.contracts import Channel, DecisionAction, LoanRequest, ManagerDecision
from loan_agents.core.masking import mask_personal_data, names_in
from loan_agents.fakes import RecordingMessenger
from loan_agents.settings import Settings
from loan_agents.wiring import build_runtime
from tests.conftest import TODAY, sample_bytes

PERSONAL = [
    "Тарас",
    "Шевченко",
    "ШЕВЧЕНКО",
    "ГРИГОРОВИЧ",
    "Мене звати",
    "067 123 45 67",
    "0671234567",
    "+380671234567",
    "TMBJJ7NE8J0123456",
    "AA1234BB",
    "base64,",
]


def _as_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"full_name": "Тарас Шевченко"}, {"full_name": "Т. Ш."}),
        ({"owner_full_name": "ШЕВЧЕНКО ТАРАС"}, {"owner_full_name": "Ш. Т."}),
        ({"lastName": "Шевченко"}, {"lastName": "Ш."}),
        ({"phone": "067 123 45 67"}, {"phone": "+38067***4567"}),
        ({"phoneNumber": "+380671234567"}, {"phoneNumber": "+38067***4567"}),
        ({"vin": "TMBJJ7NE8J0123456"}, {"vin": "TMB**********3456"}),
        ({"plate_number": "AA1234BB"}, {"plate_number": "AA****BB"}),
    ],
)
def test_known_fields_are_masked_by_name(data: dict[str, str], expected: dict[str, str]) -> None:
    assert mask_personal_data(data) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("дзвоніть на 067 123 45 67", "дзвоніть на +38067***4567"),
        ("тел. (067) 123-45-67.", "тел. +38067***4567."),
        ("+380671234567", "+38067***4567"),
        ("VIN TMBJJ7NE8J0123456", "VIN TMB**********3456"),
        ("номер AA1234BB", "номер AA****BB"),
        ("номер АА1234ВВ", "номер АА****ВВ"),  # Cyrillic letters, as on the plate itself
    ],
)
def test_identifiers_are_masked_in_any_text(text: str, expected: str) -> None:
    assert mask_personal_data({"note": text}) == {"note": expected}


@pytest.mark.parametrize(
    "harmless",
    [
        "200 000 грн на 12 місяців",
        "deal 6af0e79b04337ba0e",  # an EspoCRM id: lowercase, so not a VIN
        "12345678901234567",  # 17 digits are not a VIN either
        "SKODA OCTAVIA 2018",
    ],
)
def test_ordinary_text_is_left_alone(harmless: str) -> None:
    assert mask_personal_data(harmless) == harmless


def test_names_from_name_fields_are_masked_everywhere_in_same_data() -> None:
    state = {
        "request": {"full_name": "Тарас Шевченко"},
        "certificate": {"owner_full_name": "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ"},
        "contract_text": "Позичальник: Тарас Шевченко. Власник: ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ.",
    }
    masked = mask_personal_data(state)
    assert masked["contract_text"] == "Позичальник: Т. Ш.. Власник: Ш. Т. Г.."


def test_names_known_from_elsewhere_are_masked_too() -> None:
    assert names_in({"request": {"full_name": "Тарас Шевченко"}}) == {"Тарас", "Шевченко"}
    masked = mask_personal_data("Договір з Тарас Шевченко", known_names={"Тарас", "Шевченко"})
    assert masked == "Договір з Т. Ш."


@pytest.mark.parametrize(
    "data",
    [
        {"inbox_text": "Мене звати Тарас Шевченко"},
        [{"role": "user", "content": "Мене звати Тарас Шевченко"}],
        [{"type": "human", "content": "Мене звати Тарас Шевченко", "additional_kwargs": {}}],
    ],
    ids=["state", "chat-message", "langchain-message"],
)
def test_client_text_is_replaced_by_its_length(data: Any) -> None:
    masked = _as_text(mask_personal_data(data))
    assert "Тарас" not in masked
    assert "[client's message, 25 characters]" in masked


def test_our_prompt_text_is_kept_and_pictures_are_replaced() -> None:
    picture = "data:image/jpeg;base64," + base64.b64encode(b"x" * 3000).decode()
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Read this image."},
            {"type": "image_url", "image_url": {"url": picture}},
        ],
    }
    masked = mask_personal_data(message)
    assert masked["content"][0]["text"] == "Read this image."
    assert masked["content"][1]["image_url"]["url"] == "[image/jpeg, 2 KB, not sent]"


def test_model_json_answer_is_masked_like_any_data() -> None:
    answer = json.dumps({"full_name": "Тарас Шевченко", "amount_uah": 200000}, ensure_ascii=False)
    assert json.loads(mask_personal_data(answer)) == {"full_name": "Т. Ш.", "amount_uah": 200000}


def test_models_and_dataclasses_are_masked_as_their_fields() -> None:
    @dataclasses.dataclass
    class Interrupt:
        value: dict[str, str]

    request = LoanRequest(full_name="Тарас Шевченко", phone="+380671234567")
    masked = mask_personal_data([request, Interrupt({"card": "Клієнт: Тарас Шевченко"})])
    assert masked[0]["full_name"] == "Т. Ш."
    assert masked[0]["phone"] == "+38067***4567"
    assert masked[1] == {"value": {"card": "Клієнт: Т. Ш."}}


def test_numbers_flags_and_empty_values_pass_through() -> None:
    data = {"amount_uah": 200000, "ok": True, "rate": 0.18, "region": None, "phone": ""}
    assert mask_personal_data(data) == data


def _open(exporter: InMemorySpanExporter) -> LangfuseTracer:
    # A key of its own per client: the SDK keeps one client per public key for the whole process.
    return open_langfuse(
        "http://127.0.0.1:9",
        f"pk-lf-test-{uuid.uuid4()}",
        "sk-lf-test",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
    )


@pytest.fixture
def exported() -> Iterator[tuple[LangfuseTracer, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    tracer = _open(exporter)
    yield tracer, exporter
    tracer.shutdown()


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in (span.attributes or {}).items():
        try:
            decoded[key] = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            decoded[key] = value
    return decoded


async def test_traced_case_leaves_no_personal_data_in_any_span(
    exported: tuple[LangfuseTracer, InMemorySpanExporter], settings: Settings
) -> None:
    tracer, exporter = exported
    async with build_runtime(
        settings, fake=True, messenger=RecordingMessenger(), tracer=tracer, today=lambda: TODAY
    ) as runtime:
        service = runtime.service
        await service.start("web-1", Channel.WEB)
        for text in ("Мене звати Тарас Шевченко", "067 123 45 67", "200 тис грн на 12 місяців"):
            await service.handle_text("web-1", Channel.WEB, text)
        await service.handle_text("web-1", Channel.WEB, "Шкода Октавія 2018 року")
        await service.handle_document(
            "web-1", Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )
        await service.decide("web-1", ManagerDecision(action=DecisionAction.APPROVE, manager="m"))
    tracer.shutdown()  # sends what is still queued

    spans = exporter.get_finished_spans()
    roots = [span for span in spans if span.parent is None]
    assert [root.name for root in roots] == [
        "start",
        "message",
        "message",
        "message",
        "message",
        "document",
        "decision",
    ]
    assert len({span.context.trace_id for span in spans}) == len(roots)
    assert {span.attributes.get("session.id") for span in spans} == {"web-1"}
    everything = _as_text([_attributes(span) for span in spans])
    for value in PERSONAL:
        assert value not in everything, value
    assert "+38067***4567" in everything  # the case is in the trace, masked
    # The approval node pauses the graph with an interrupt whose text holds the whole card.
    statuses = {span.attributes.get("langfuse.observation.status_message") for span in spans}
    assert statuses - {None} == {"paused: waiting for input"}


async def test_concurrent_cases_keep_separate_traces(
    exported: tuple[LangfuseTracer, InMemorySpanExporter], settings: Settings
) -> None:
    tracer, exporter = exported
    async with build_runtime(
        settings, fake=True, messenger=RecordingMessenger(), tracer=tracer, today=lambda: TODAY
    ) as runtime:
        service = runtime.service

        async def apply(case_id: str, name: str) -> None:
            await service.start(case_id, Channel.WEB)
            await service.handle_text(case_id, Channel.WEB, f"Мене звати {name}")
            await service.handle_text(case_id, Channel.WEB, "067 123 45 67")

        await asyncio.gather(apply("web-a", "Тарас Шевченко"), apply("web-b", "Олена Коваленко"))
    tracer.shutdown()

    sessions: dict[int, set[str]] = defaultdict(set)
    for span in exporter.get_finished_spans():
        sessions[span.context.trace_id].add(span.attributes.get("session.id"))
    assert sorted(sessions.values(), key=sorted) == [{"web-a"}] * 3 + [{"web-b"}] * 3
    everything = _as_text([_attributes(span) for span in exporter.get_finished_spans()])
    for value in ("Тарас", "Шевченко", "Олена", "Коваленко"):
        assert value not in everything, value


async def test_model_call_is_traced_with_its_answer_and_without_picture(
    exported: tuple[LangfuseTracer, InMemorySpanExporter],
) -> None:
    tracer, exporter = exported
    model = FakeListChatModel(responses=['{"owner_full_name": "ШЕВЧЕНКО ТАРАС"}'])
    picture = (
        "data:image/jpeg;base64," + base64.b64encode(sample_bytes("certificate-ok.png")).decode()
    )
    messages = [
        SystemMessage("Read the certificate."),
        HumanMessage(
            content=[
                {"type": "text", "text": "Read this image."},
                {"type": "image_url", "image_url": {"url": picture}},
            ]
        ),
    ]
    await model.ainvoke(messages, tracer.run_config("tg-1", "document"))
    tracer.shutdown()

    (generation,) = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("langfuse.observation.type") == "generation"
    ]
    attributes = _attributes(generation)
    sent = _as_text(attributes["langfuse.observation.input"])
    assert "base64," not in sent
    assert "not sent]" in sent
    assert "Ш. Т." in _as_text(attributes["langfuse.observation.output"])
    assert "ШЕВЧЕНКО" not in _as_text(attributes)


def test_sdk_is_never_allowed_to_upload_pictures(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK would upload them before the mask runs.
    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "true")
    _open(InMemorySpanExporter()).shutdown()
    assert os.environ["LANGFUSE_MEDIA_UPLOAD_ENABLED"] == "false"


async def test_keys_in_settings_turn_tracing_on(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    traced = settings.model_copy(
        update={
            "langfuse_base_url": "http://127.0.0.1:9",
            "langfuse_public_key": f"pk-lf-test-{uuid.uuid4()}",
            "langfuse_secret_key": SecretStr("sk-lf-test"),
        }
    )
    with caplog.at_level(logging.INFO, logger="loan_agents.wiring"):
        async with build_runtime(traced, fake=True):
            pass
    assert "event=tracing url=http://127.0.0.1:9" in caplog.text


PROJECT = {
    "id": "p1",
    "name": "loan-intake-agents",
    "organization": {"id": "o1", "name": "Loan intake"},
    "metadata": {},
}


def _langfuse_api(status: int) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/projects"
        if status != 200:
            return httpx.Response(status, json={"message": "Invalid credentials"})
        return httpx.Response(200, json={"data": [PROJECT]})

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_check_keys_reports_accepted_keys() -> None:
    key = f"pk-lf-{uuid.uuid4()}"
    result = check_keys("http://lf.test", key, "sk-lf-x", http=_langfuse_api(200))
    assert result == "http://lf.test: keys accepted"


def test_check_keys_says_when_keys_are_wrong() -> None:
    key = f"pk-lf-{uuid.uuid4()}"
    with pytest.raises(RuntimeError, match=r"not accepted \(HTTP 401\)"):
        check_keys("http://lf.test", key, "sk-lf-x", http=_langfuse_api(401))


def test_check_keys_says_how_to_start_langfuse_when_it_is_not_running() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    with pytest.raises(RuntimeError, match="docker compose --profile tracing up -d"):
        check_keys("http://lf.test", f"pk-lf-{uuid.uuid4()}", "sk-lf-x", http=http)


async def test_doctor_skips_tracing_without_keys(settings: Settings) -> None:
    assert (await _check_langfuse(settings)).startswith("skipped")


def test_tracing_keys_fills_missing_values_and_keeps_rest(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LANGFUSE_PUBLIC_KEY=\nLANGFUSE_SECRET_KEY=\nLOG_LEVEL=INFO\n")
    monkeypatch.setattr("loan_agents.cli.ENV_FILE", env_file)
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: settings)
    assert main(["tracing-keys", "--write-env"]) == 0
    written = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert written["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-")
    assert written["LANGFUSE_SECRET_KEY"].startswith("sk-lf-")
    assert len(written["LANGFUSE_ADMIN_PASSWORD"]) >= 20
    assert written["LOG_LEVEL"] == "INFO"


def test_tracing_keys_never_replaces_what_langfuse_already_has(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    configured = settings.model_copy(
        update={
            "langfuse_public_key": "pk-lf-1",
            "langfuse_secret_key": SecretStr("sk-lf-1"),
            "langfuse_admin_password": SecretStr("secret"),
        }
    )
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: configured)
    assert main(["tracing-keys"]) == 0
    assert "set already" in capsys.readouterr().out
