"""Model calls through LangChain, so the provider is one setting: LLM_MODEL=ollama:qwen3-vl:8b.

With Ollama the schema is enforced by constrained decoding (format = JSON schema) and thinking is
switched off: these are extraction tasks, and reasoning tokens only add latency.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import Any, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from PIL import Image, ImageOps
from pydantic import BaseModel, ValidationError

from loan_agents.contracts import CertificateReading, RequestExtraction
from loan_agents.ports import ExtractorError

log = logging.getLogger(__name__)

MAX_IMAGE_SIDE_PX = 1600
_Schema = TypeVar("_Schema", bound=BaseModel)

REQUEST_PROMPT = """\
You read one message from a client applying for a loan secured by a car. The message is usually
in Ukrainian, sometimes in Russian or English. Extract only facts the client states in this
message. Use null for anything not stated; never guess or fill in defaults.

- full_name: the client's own name exactly as written.
- phone: the phone number as written.
- amount_uah: the loan amount in hryvnias as an integer ("200 тис" -> 200000).
- term_months: the loan term in months ("рік" -> 12, "півтора року" -> 18).
- car_make, car_model: the car's make and model in Latin script, as on registration documents
  ("Шкода Октавія" -> SKODA, OCTAVIA).
- car_year: the car's year of manufacture.
- region: an oblast or city the client names.
- question_topic: if the client asks a question, its topic: rate (interest, payments),
  documents (what papers are needed), timing (how long, when) or other. Otherwise null.
"""

CERTIFICATE_PROMPT = """\
The image should show a Ukrainian vehicle registration certificate (свідоцтво про реєстрацію
транспортного засобу, also called техпаспорт). A sample stamped ЗРАЗОК counts as a certificate.
Read the printed fields exactly as printed. Use null for any field you cannot read with
certainty; never guess characters.

- is_vehicle_certificate: true only if the image is such a certificate.
- plate_number: the registration number (field A), e.g. AA1234BB.
- vin: the 17-character identification number (field E).
- make, model: fields D.1 and D.3.
- year: the year of manufacture (field B).
- owner_full_name: the owner's surname, name and patronymic (field C.1).
"""


def to_jpeg(image: bytes, max_side_px: int = MAX_IMAGE_SIDE_PX) -> bytes:
    """Re-encodes to JPEG: applies phone rotation, strips metadata, caps the size."""
    with Image.open(io.BytesIO(image)) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGB")
    picture.thumbnail((max_side_px, max_side_px))
    output = io.BytesIO()
    picture.save(output, format="JPEG", quality=90)
    return output.getvalue()


class LangChainExtractor:
    def __init__(self, model_spec: str, *, ollama_base_url: str, timeout_s: float) -> None:
        provider, _, model_name = model_spec.partition(":")
        if not provider or not model_name:
            raise ValueError(f"model spec {model_spec!r} must look like provider:model")
        chat_kwargs: dict[str, Any] = {"temperature": 0}
        structured_kwargs: dict[str, Any] = {}
        if provider == "ollama":
            chat_kwargs |= {
                "base_url": ollama_base_url,
                "reasoning": False,
                "client_kwargs": {"timeout": timeout_s},
            }
            structured_kwargs = {"method": "json_schema"}
        chat = init_chat_model(model_spec, **chat_kwargs)
        self._request_reader = chat.with_structured_output(RequestExtraction, **structured_kwargs)
        self._certificate_reader = chat.with_structured_output(
            CertificateReading, **structured_kwargs
        )
        self._model_spec = model_spec
        self._timeout_s = timeout_s

    async def _invoke(
        self,
        runnable: Runnable,
        messages: list[BaseMessage],
        schema: type[_Schema],
        task: str,
    ) -> _Schema:
        started = time.monotonic()
        try:
            result = await asyncio.wait_for(runnable.ainvoke(messages), self._timeout_s)
        except TimeoutError as error:
            raise ExtractorError(
                f"{self._model_spec}: no answer to {task} within {self._timeout_s:.0f} s"
            ) from error
        except (OutputParserException, ValidationError) as error:
            # These errors quote the model's answer, which holds the client's personal data, and
            # the service logs failed turns with their cause. LOG_LEVEL=DEBUG still shows it.
            log.debug("step=llm task=%s event=answer_not_in_schema error=%s", task, error)
            raise ExtractorError(
                f"{self._model_spec}: {task} answer did not match the schema"
            ) from None
        except Exception as error:
            raise ExtractorError(
                f"{self._model_spec}: {task} failed: {type(error).__name__}: {error}"
            ) from error
        if not isinstance(result, schema):
            raise ExtractorError(
                f"{self._model_spec}: {task} returned {type(result).__name__}, "
                f"expected {schema.__name__}"
            )
        elapsed_ms = (time.monotonic() - started) * 1000
        log.info("step=llm task=%s model=%s ms=%.0f", task, self._model_spec, elapsed_ms)
        return result

    async def extract_request(self, text: str) -> RequestExtraction:
        messages = [SystemMessage(REQUEST_PROMPT), HumanMessage(text)]
        return await self._invoke(
            self._request_reader, messages, RequestExtraction, "extract_request"
        )

    async def read_certificate(self, image: bytes, mime_type: str) -> CertificateReading:
        encoded = base64.b64encode(to_jpeg(image)).decode("ascii")
        content = [
            {"type": "text", "text": "Read this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ]
        messages = [SystemMessage(CERTIFICATE_PROMPT), HumanMessage(content=content)]
        return await self._invoke(
            self._certificate_reader, messages, CertificateReading, "read_certificate"
        )
