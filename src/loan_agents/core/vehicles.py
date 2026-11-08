"""Normalising car makes, models, plates and VINs to the forms printed on certificates."""

from __future__ import annotations

import re

MIN_CAR_YEAR = 1950

# Ukrainian plates use only letters that look the same in Cyrillic and Latin.
_PLATE_CYRILLIC_TO_LATIN = str.maketrans("АВЕІКМНОРСТХ", "ABEIKMHOPCTX")
_PLATE_PATTERN = re.compile(r"^[A-Z]{2}\d{4}[A-Z]{2}$")
# ISO 3779: 17 characters, letters I, O and Q are never used.
_VIN_PATTERN = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")

_MAKE_ALIASES = {
    "ШКОДА": "SKODA",
    "ŠKODA": "SKODA",
    "ТОЙОТА": "TOYOTA",
    "ФОЛЬКСВАГЕН": "VOLKSWAGEN",
    "VW": "VOLKSWAGEN",
    "РЕНО": "RENAULT",
    "ХЮНДАЙ": "HYUNDAI",
    "ХЕНДАЙ": "HYUNDAI",
    "КІА": "KIA",
    "КИА": "KIA",
    "НІСАН": "NISSAN",
    "НІССАН": "NISSAN",
    "ФОРД": "FORD",
    "БМВ": "BMW",
    "АУДІ": "AUDI",
    "МЕРСЕДЕС": "MERCEDES-BENZ",
    "MERCEDES": "MERCEDES-BENZ",
    "ОПЕЛЬ": "OPEL",
    "ПЕЖО": "PEUGEOT",
    "МАЗДА": "MAZDA",
    "ХОНДА": "HONDA",
}

_MODEL_ALIASES = {
    "ОКТАВІЯ": "OCTAVIA",
    "ОКТАВИЯ": "OCTAVIA",
    "КАМРІ": "CAMRY",
    "КАМРИ": "CAMRY",
    "РАВ4": "RAV4",
    "RAV-4": "RAV4",
    "ПАССАТ": "PASSAT",
    "ПАСАТ": "PASSAT",
    "ЛОГАН": "LOGAN",
    "ТУКСОН": "TUCSON",
    "ТУСАН": "TUCSON",
    "СПОРТЕЙДЖ": "SPORTAGE",
    "КАШКАЙ": "QASHQAI",
    "ФОКУС": "FOCUS",
    "Х5": "X5",
}


def _key(raw: str) -> str:
    return " ".join(raw.upper().split())


def normalize_make(raw: str) -> str:
    key = _key(raw)
    return _MAKE_ALIASES.get(key, key)


def normalize_model(raw: str) -> str:
    key = _key(raw)
    return _MODEL_ALIASES.get(key, key)


def normalize_plate(raw: str) -> str:
    compact = "".join(raw.upper().split()).replace("-", "")
    return compact.translate(_PLATE_CYRILLIC_TO_LATIN)


def is_valid_plate(plate: str) -> bool:
    return bool(_PLATE_PATTERN.match(plate))


def normalize_vin(raw: str) -> str:
    return "".join(raw.upper().split())


def is_valid_vin(vin: str) -> bool:
    return bool(_VIN_PATTERN.match(vin))
