"""The LangChain adapter against a stand-in Ollama server on a local port.

The request building and the answer parsing run through the real langchain-ollama and ollama client
code; only the model is replaced, by canned answers.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from loan_agents.adapters.llm_langchain import MAX_IMAGE_SIDE_PX, LangChainExtractor
from loan_agents.contracts import CertificateReading, QuestionTopic, RequestExtraction
from loan_agents.ports import ExtractorError
from loan_agents.samples import SAMPLES
from tests.conftest import sample_bytes

MODEL = "ollama:qwen3-vl:8b"
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

    def build_app(self) -> FastAPI:
        app = FastAPI()

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
                {"model": body["model"], "message": {"role": "assistant", "content": ""}},
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


async def test_a_message_is_extracted_with_the_schema_enforced(ollama) -> None:
    fake, base_url = ollama
    result = await _extractor(base_url).extract_request("Мене звати Тарас Шевченко ...")
    assert result == RequestExtraction.model_validate(REQUEST_ANSWER)
    assert result.question_topic == QuestionTopic.RATE
    sent = fake.requests[0]
    assert sent["model"] == "qwen3-vl:8b"
    assert sent["think"] is False
    assert sent["options"]["temperature"] == 0
    assert set(sent["format"]["properties"]) == set(RequestExtraction.model_fields)


async def test_a_certificate_goes_out_as_a_downscaled_jpeg(ollama) -> None:
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
        ("", (404, "model 'qwen3-vl:8b' not found")),
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
async def test_a_rejected_answer_keeps_client_data_out_of_the_error(ollama, content) -> None:
    fake, base_url = ollama
    fake.content = content
    with pytest.raises(ExtractorError, match="did not match the schema") as caught:
        await _extractor(base_url).extract_request("Привіт")
    assert "067" not in str(caught.value)
    assert caught.value.__cause__ is None and caught.value.__suppress_context__


async def test_a_slow_model_times_out(ollama) -> None:
    fake, base_url = ollama
    fake.delay_s = 1.0
    with pytest.raises(ExtractorError, match="extract_request"):
        await _extractor(base_url, timeout_s=0.2).extract_request("Привіт")


async def test_ollama_not_running_is_an_extractor_error() -> None:
    with pytest.raises(ExtractorError, match="extract_request"):
        await _extractor("http://127.0.0.1:9", timeout_s=5).extract_request("Привіт")
