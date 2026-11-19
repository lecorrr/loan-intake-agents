"""Masking personal data everywhere but the CRM, which keeps the full values."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from pydantic import BaseModel

from loan_agents.contracts import LoanRequest, VehicleCertificate
from loan_agents.core.intake import normalize_phone


def mask_phone(phone: str) -> str:
    if len(phone) < 10:
        return "***"
    return f"{phone[:6]}***{phone[-4:]}"


def mask_vin(vin: str) -> str:
    if len(vin) < 8:
        return "***"
    return f"{vin[:3]}{'*' * (len(vin) - 7)}{vin[-4:]}"


def mask_plate(plate: str) -> str:
    if len(plate) < 5:
        return "***"
    return f"{plate[:2]}{'*' * (len(plate) - 4)}{plate[-2:]}"


def mask_name(name: str) -> str:
    return " ".join(f"{token[0]}." for token in name.split() if token)


def masked_request(request: LoanRequest) -> dict[str, str | int | None]:
    values = request.model_dump()
    if request.full_name:
        values["full_name"] = mask_name(request.full_name)
    if request.phone:
        values["phone"] = mask_phone(request.phone)
    return values


def masked_certificate(certificate: VehicleCertificate) -> dict[str, str | int]:
    values = certificate.model_dump()
    values["vin"] = mask_vin(certificate.vin)
    values["plate_number"] = mask_plate(certificate.plate_number)
    values["owner_full_name"] = mask_name(certificate.owner_full_name)
    return values


_NAME_FIELDS = frozenset(
    {"full_name", "owner_full_name", "first_name", "last_name", "middle_name"}
    | {"firstName", "lastName", "middleName"}
)
_FIELD_MASKS: dict[str, Callable[[str], str]] = {
    "phone": lambda value: mask_phone(normalize_phone(value) or value),
    "phoneNumber": lambda value: mask_phone(normalize_phone(value) or value),
    "vin": mask_vin,
    "plate_number": mask_plate,
} | {field: mask_name for field in _NAME_FIELDS}

_IMAGE_DATA = re.compile(r"data:(image/[\w.+-]+);base64,[A-Za-z0-9+/=]+")
_PHONE_IN_TEXT = re.compile(
    r"(?<![\d+])(?:\+?38[\s-]*)?\(?0\d{2}\)?[\s-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"
)
_VIN_IN_TEXT = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
# Ukrainian plates use the Cyrillic letters that look Latin; people type either.
_PLATE_IN_TEXT = re.compile(r"\b[A-ZАВЕІКМНОРСТХ]{2} ?\d{4} ?[A-ZАВЕІКМНОРСТХ]{2}\b")


def mask_personal_data(value: Any, known_names: Iterable[str] = ()) -> Any:
    """Copy of JSON-like data with the client's personal data masked.

    Fields are masked by name; phones, VINs and plates by shape; names wherever they appear
    (from name fields in the same data or known_names). Data-URL pictures become a note. The
    client's own words (`inbox_text`, plain user messages) are replaced by their length.
    """
    names = _names_pattern(names_in(value) | {name for name in known_names if len(name) > 1})
    return _masked(value, names)


def names_in(value: Any) -> set[str]:
    """The words of every name field in the data, for masking those names elsewhere."""
    return _collect_names(value, set())


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return value


def _parsed_json_object(text: str) -> dict[str, Any] | None:
    if not text.lstrip().startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _collect_names(value: Any, names: set[str]) -> set[str]:
    value = _plain(value)
    if isinstance(value, str):
        parsed = _parsed_json_object(value)
        if parsed is not None:
            _collect_names(parsed, names)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if key in _NAME_FIELDS and isinstance(item, str):
                names.update(token for token in item.split() if len(token) > 1)
            else:
                _collect_names(item, names)
    elif isinstance(value, list | tuple | set):
        for item in value:
            _collect_names(item, names)
    return names


def _names_pattern(names: set[str]) -> re.Pattern[str] | None:
    if not names:
        return None
    # Longest first, so a name that contains another is masked whole.
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


def _masked(value: Any, names: re.Pattern[str] | None) -> Any:
    value = _plain(value)
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _masked_text(value, names)
    if isinstance(value, Mapping):
        masked = {key: _masked_field(str(key), item, names) for key, item in value.items()}
        if _is_client_message(value):
            masked["content"] = _client_words(value["content"])
        return masked
    if isinstance(value, list | tuple | set):
        return [_masked(item, names) for item in value]
    if isinstance(value, bytes):
        return f"[{len(value)} bytes]"
    # Anything else would reach the outside as its text anyway.
    return _masked_text(str(value), names)


def _masked_field(key: str, value: Any, names: re.Pattern[str] | None) -> Any:
    mask = _FIELD_MASKS.get(key)
    if mask is not None and isinstance(value, str) and value:
        return mask(value)
    if key == "inbox_text" and isinstance(value, str):
        return _client_words(value)
    return _masked(value, names)


def _client_words(text: str) -> str:
    return f"[client's message, {len(text)} characters]"


def _is_client_message(message: Mapping[str, Any]) -> bool:
    # As a chat message ({"role": "user"}) or as a LangChain message ({"type": "human"}).
    sender = message.get("role") or message.get("type")
    return sender in {"user", "human"} and isinstance(message.get("content"), str)


def _masked_text(text: str, names: re.Pattern[str] | None) -> str:
    parsed = _parsed_json_object(text)
    if parsed is not None:
        return json.dumps(_masked(parsed, names), ensure_ascii=False)
    text = _IMAGE_DATA.sub(_image_note, text)
    if names is not None:
        text = names.sub(lambda match: f"{match.group(0)[0]}.", text)
    text = _PHONE_IN_TEXT.sub(_masked_phone, text)
    text = _VIN_IN_TEXT.sub(_masked_vin, text)
    return _PLATE_IN_TEXT.sub(lambda match: mask_plate(match.group(0)), text)


def _image_note(match: re.Match[str]) -> str:
    size_kb = len(match.group(0)) * 3 // 4 // 1024
    return f"[{match.group(1)}, {size_kb} KB, not sent]"


def _masked_phone(match: re.Match[str]) -> str:
    normalized = normalize_phone(match.group(0))
    return mask_phone(normalized) if normalized else match.group(0)


def _masked_vin(match: re.Match[str]) -> str:
    vin = match.group(0)
    # A VIN mixes letters and digits; an all-digit or all-letter run is something else.
    if vin.isdigit() or vin.isalpha():
        return vin
    return mask_vin(vin)
