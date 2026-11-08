"""A rule-based extractor: regexes for text, embedded ground truth for sample images.

It understands the demo conversation and the generated samples, nothing more. Photos without
embedded ground truth (for example real Telegram photos) read as "not a certificate".
"""

from __future__ import annotations

import io
import json
import re

from PIL import Image, UnidentifiedImageError

from loan_agents.contracts import CertificateReading, RequestExtraction

TRUTH_KEY = "loan_agents_truth"

_KNOWN_MAKES = {
    "SKODA": "SKODA",
    "ШКОДА": "SKODA",
    "TOYOTA": "TOYOTA",
    "ТОЙОТА": "TOYOTA",
    "RENAULT": "RENAULT",
    "РЕНО": "RENAULT",
    "VOLKSWAGEN": "VOLKSWAGEN",
}
_KNOWN_MODELS = {
    "OCTAVIA": "OCTAVIA",
    "ОКТАВІЯ": "OCTAVIA",
    "CAMRY": "CAMRY",
    "КАМРІ": "CAMRY",
    "LOGAN": "LOGAN",
    "ЛОГАН": "LOGAN",
    "PASSAT": "PASSAT",
}
_WORD = re.compile(r"[\w'’ʼ-]+")
_NAME_WORD = r"[А-ЯІЇЄҐ][а-яіїєґ'’ʼ-]+"
_NAME = re.compile(
    rf"(?:мене звати|я\s*[—-]|моє ім'я)\s+({_NAME_WORD}(?:\s+{_NAME_WORD}){{1,2}})",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?:\+?38)?0\d{2}[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
_AMOUNT_THOUSANDS = re.compile(r"(\d{1,4})\s*(?:тис|тисяч)", re.IGNORECASE)
_AMOUNT_UAH = re.compile(r"(\d[\d\s]{3,})\s*(?:грн|гривень)", re.IGNORECASE)
_TERM = re.compile(r"(\d{1,3})\s*(?:міс|місяц)", re.IGNORECASE)
_YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\s*(?:р\.|року|рік)", re.IGNORECASE)


def _question_topic(text: str) -> str | None:
    lowered = text.lower()
    if "?" not in text:
        return None
    if "ставк" in lowered or "відсот" in lowered:
        return "rate"
    if "документ" in lowered:
        return "documents"
    if "скільки часу" in lowered or "коли" in lowered or "як довго" in lowered:
        return "timing"
    return "other"


class FakeExtractor:
    def __init__(self) -> None:
        self.request_calls = 0
        self.certificate_calls = 0

    async def extract_request(self, text: str) -> RequestExtraction:
        self.request_calls += 1
        values: dict[str, object] = {}
        if name := _NAME.search(text):
            values["full_name"] = name.group(1)
        if phone := _PHONE.search(text):
            values["phone"] = phone.group(0)
        if amount := _AMOUNT_THOUSANDS.search(text):
            values["amount_uah"] = int(amount.group(1)) * 1000
        elif amount := _AMOUNT_UAH.search(text):
            values["amount_uah"] = int(re.sub(r"\s", "", amount.group(1)))
        if term := _TERM.search(text):
            values["term_months"] = int(term.group(1))
        if year := _YEAR.search(text):
            values["car_year"] = int(year.group(1))
        words = [word.upper() for word in _WORD.findall(text)]
        for word in words:
            if word in _KNOWN_MAKES:
                values["car_make"] = _KNOWN_MAKES[word]
            if word in _KNOWN_MODELS:
                values["car_model"] = _KNOWN_MODELS[word]
        values["question_topic"] = _question_topic(text)
        return RequestExtraction.model_validate(values)

    async def read_certificate(self, image: bytes, mime_type: str) -> CertificateReading:
        self.certificate_calls += 1
        try:
            with Image.open(io.BytesIO(image)) as picture:
                truth = picture.info.get(TRUTH_KEY)
        except UnidentifiedImageError:
            truth = None
        if truth is None:
            return CertificateReading(is_vehicle_certificate=False)
        return CertificateReading.model_validate(json.loads(truth))
