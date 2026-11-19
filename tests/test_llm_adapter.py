"""LangChain adapter against a fake Ollama server on a local port.

Requests and parsing go through the real langchain-ollama and ollama code; only the answers
are canned.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from loan_agents.adapters.llm_langchain import MAX_IMAGE_SIDE_PX, LangChainExtractor
from loan_agents.cli import CheckWarning, _check_ollama
from loan_agents.contracts import CertificateReading, QuestionTopic, RequestExtraction
from loan_agents.ports import ExtractorError
from loan_agents.samples import SAMPLES
from loan_agents.settings import Settings
from tests.conftest import sample_bytes

MODEL = "ollama:qwen3-vl:8b-instruct"
REQUEST_ANSWER = {
    "full_name": "Тарас Шевченко",
    "phone": "067 123 45 67",
    "amount_uah": 200000,
    "term_months": 12,
    "car_make": "SKODA",
    "car_model": "OCTAVIA",
    "car_year": 2018,
    "region": None,
    "question_topic": "rate",
}


class FakeOllama:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.content = json.dumps(REQUEST_ANSWER)
        self.delay_s = 0.0
        self.error: tuple[int, str] | None = None
        self.version = "0.40.15"
        self.pulled = ["qwen3-vl:8b-instruct"]
        self.capabilities = ["completion", "vision", "tools"]

    def build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/api/version")
        async def version() -> dict[str, str]:
            return {"version": self.version}

        @app.get("/api/tags")
        async def tags() -> dict[str, Any]:
            return {"models": [{"name": name} for name in self.pulled]}

        @app.post("/api/show")
        async def show() -> dict[str, Any]:
            return {"capabilities": self.capabilities}

        @app.post("/api/chat")
        async def chat(request: Request) -> Any:
            body = await request.json()
            self.requests.append(body)
            await asyncio.sleep(self.delay_s)
            if self.error is not None:
                status, message = self.error
                return JSONResponse({"error": message}, status_code=status)
            chunks = [
                {"model": body["model"], "message": {"role": "assistant", "content": self.content}},
                {
                    "model": body["model"],
                    "message": {"role": "assistant", "content": ""},
                    "prompt_eval_count": 1261,
                    "eval_count": 97,
                },
            ]
            lines = [
                json.dumps(
                    chunk
                    | {"created_at": "2026-11-08T10:00:00Z", "done": index == 1}
                    | ({"done_reason": "stop"} if index == 1 else {})
                )
                + "\n"
                for index, chunk in enumerate(chunks)
            ]
            return StreamingResponse(iter(lines), media_type="application/x-ndjson")

        return app


@pytest.fixture
async def ollama() -> AsyncIterator[tuple[FakeOllama, str]]:
    fake = FakeOllama()
    server = uvicorn.Server(
        uvicorn.Config(fake.build_app(), host="127.0.0.1", port=0, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    async with asyncio.timeout(10):  # uvicorn has no startup event to await, so poll
        while not server.started:
            if task.done():
                task.result()  # re-raises the startup error
                raise RuntimeError("the stand-in Ollama server stopped during startup")
            await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield fake, f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task


def _extractor(base_url: str, timeout_s: float = 10) -> LangChainExtractor:
    return LangChainExtractor(MODEL, ollama_base_url=base_url, timeout_s=timeout_s)


async def test_message_is_extracted_with_schema_enforced(ollama) -> None:
    fake, base_url = ollama
    result = await _extractor(base_url).extract_request("Мене звати Тарас Шевченко ...")
    assert result == RequestExtraction.model_validate(REQUEST_ANSWER)
    assert result.question_topic == QuestionTopic.RATE
    sent = fake.requests[0]
    assert sent["model"] == "qwen3-vl:8b-instruct"
    assert sent["think"] is False
    assert sent["options"]["temperature"] == 0
    assert set(sent["format"]["properties"]) == set(RequestExtraction.model_fields)


async def test_certificate_goes_out_as_downscaled_jpeg(ollama) -> None:
    fake, base_url = ollama
    truth = next(sample.truth for sample in SAMPLES if sample.file_name == "certificate-ok.png")
    fake.content = json.dumps(truth)
    reading = await _extractor(base_url).read_certificate(
        sample_bytes("certificate-ok.png"), "image/png"
    )
    assert reading == CertificateReading.model_validate(truth)
    user_message = fake.requests[0]["messages"][-1]
    (image,) = user_message["images"]
    with Image.open(io.BytesIO(base64.b64decode(image))) as sent:
        assert sent.format == "JPEG"
        assert max(sent.size) <= MAX_IMAGE_SIDE_PX


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not json at all", None),
        (json.dumps({"amount_uah": "lots"}), None),
        ("", (404, "model 'qwen3-vl:8b-instruct' not found")),
    ],
    ids=["not-json", "wrong-types", "model-missing"],
)
async def test_unusable_answers_become_extractor_errors(ollama, content, error) -> None:
    fake, base_url = ollama
    fake.content, fake.error = content, error
    with pytest.raises(ExtractorError, match="extract_request"):
        await _extractor(base_url).extract_request("Привіт")


@pytest.mark.parametrize(
    "content",
    [
        json.dumps({"phone": "067 123 45 67", "amount_uah": "lots"}),
        "Тарас Шевченко, 067 123 45 67",
    ],
    ids=["wrong-types", "not-json"],
)
async def test_rejected_answer_keeps_client_data_out_of_error(ollama, content) -> None:
    fake, base_url = ollama
    fake.content = content
    with pytest.raises(ExtractorError, match="did not match the schema") as caught:
        await _extractor(base_url).extract_request("Привіт")
    # The service logs a failed turn with its whole exception chain.
    error: BaseException | None = caught.value
    while error is not None:
        assert "067" not in str(error)
        error = error.__cause__ or error.__context__


async def test_every_call_logs_and_reports_its_token_counts(
    ollama, caplog: pytest.LogCaptureFixture
) -> None:
    _fake, base_url = ollama
    counted: list[tuple[str, int | None, int | None]] = []
    extractor = LangChainExtractor(
        MODEL,
        ollama_base_url=base_url,
        timeout_s=10,
        on_usage=lambda task, sent, received: counted.append((task, sent, received)),
    )
    with caplog.at_level(logging.INFO, logger="loan_agents.adapters.llm_langchain"):
        await extractor.extract_request("Привіт")
    assert "input_tokens=1261 output_tokens=97" in caplog.text
    assert counted == [("extract_request", 1261, 97)]


def _doctor_settings(settings: Settings, base_url: str, model: str = MODEL) -> Settings:
    return settings.model_copy(update={"ollama_base_url": base_url, "llm_model": model})


async def test_doctor_accepts_pulled_instruct_build(ollama, settings: Settings) -> None:
    _fake, base_url = ollama
    assert await _check_ollama(_doctor_settings(settings, base_url)) == (
        "Ollama 0.40.15 has qwen3-vl:8b-instruct"
    )


async def test_doctor_warns_about_reasoning_builds(ollama, settings: Settings) -> None:
    fake, base_url = ollama
    fake.pulled = ["qwen3-vl:8b"]
    fake.capabilities = ["completion", "vision", "tools", "thinking"]
    with pytest.raises(CheckWarning, match="reasoning build"):
        await _check_ollama(_doctor_settings(settings, base_url, "ollama:qwen3-vl:8b"))


@pytest.mark.parametrize(
    ("version", "pulled", "message"),
    [
        ("0.12.6", ["qwen3-vl:8b-instruct"], "0.12.7 or newer"),
        ("0.40.15", ["qwen3-vl:8b"], "not pulled; run: ollama pull qwen3-vl:8b-instruct"),
    ],
    ids=["old-ollama", "not-pulled"],
)
async def test_doctor_fails_when_model_cannot_run(
    ollama, settings: Settings, version: str, pulled: list[str], message: str
) -> None:
    fake, base_url = ollama
    fake.version, fake.pulled = version, pulled
    with pytest.raises(RuntimeError, match=message):
        await _check_ollama(_doctor_settings(settings, base_url))


async def test_slow_model_times_out(ollama) -> None:
    fake, base_url = ollama
    fake.delay_s = 1.0
    with pytest.raises(ExtractorError, match="extract_request"):
        await _extractor(base_url, timeout_s=0.2).extract_request("Привіт")


async def test_ollama_not_running_is_extractor_error() -> None:
    with pytest.raises(ExtractorError, match="extract_request"):
        await _extractor("http://127.0.0.1:9", timeout_s=5).extract_request("Привіт")
