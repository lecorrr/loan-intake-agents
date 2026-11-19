"""Normalising car makes, models, plates and VINs to the forms printed on certificates."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

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

KNOWN_MAKES = frozenset(_MAKE_ALIASES.values())
KNOWN_MODELS = frozenset(_MODEL_ALIASES.values())

# The model sometimes spells car names the way they sound in Ukrainian (KASHKAI, OKTAVIYA,
# TOIOTA). These rules give such spellings the same key; broader ones would merge different
# cars (X -> KS would match Mazda's CX-5 with BMW's X5).
_SOUND_RULES = (
    (re.compile(r"C(?=[EIY])"), "S"),
    (re.compile(r"[CQ]"), "K"),
    (re.compile(r"KH"), "H"),
    (re.compile(r"SH"), "S"),
    (re.compile(r"W"), "V"),
    (re.compile(r"[YJ]"), "I"),
    (re.compile(r"(.)\1+"), r"\1"),
)


def _sound_key(name: str) -> str:
    """'Kashkai' and 'QASHQAI' -> 'KASKAI'. A name with non-Latin letters gets '', never a match."""
    letters = unicodedata.normalize("NFKD", name.upper())
    letters = "".join(char for char in letters if not unicodedata.combining(char))
    if any(char.isalpha() and not "A" <= char <= "Z" for char in letters):
        return ""
    # Spaces and hyphens go last, so no rule reaches across words: TUCSON-N stays apart from TUCSON.
    for pattern, replacement in _SOUND_RULES:
        letters = pattern.sub(replacement, letters)
    return re.sub(r"[^A-Z0-9]", "", letters)


def _index_by_sound(names: Iterable[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for name in sorted(names):
        key = _sound_key(name)
        if not key:
            raise ValueError(f"car name {name!r} has no sound key")
        if key in index:
            raise ValueError(f"car names {index[key]} and {name} have the same sound key {key}")
        index[key] = name
    return index


_MAKES_BY_SOUND = _index_by_sound(KNOWN_MAKES)
_MODELS_BY_SOUND = _index_by_sound(KNOWN_MODELS)


def _key(raw: str) -> str:
    return " ".join(raw.upper().split())


def normalize_make(raw: str) -> str:
    """The make as certificates print it: 'Шкода' and 'Skoda' -> 'SKODA'."""
    key = _key(raw)
    return _MAKE_ALIASES.get(key, key)


def normalize_model(raw: str) -> str:
    """The model as certificates print it: 'Октавія' and 'Octavia' -> 'OCTAVIA'."""
    key = _key(raw)
    return _MODEL_ALIASES.get(key, key)


# Client input only: a misread certificate stays a mismatch for the manager.
def normalize_client_make(raw: str) -> str:
    """normalize_make, which also knows a make spelled as it sounds: 'SHKODA' -> 'SKODA'."""
    make = normalize_make(raw)
    return _MAKES_BY_SOUND.get(_sound_key(make), make)


def normalize_client_model(raw: str) -> str:
    """normalize_model, which also knows a model spelled as it sounds: 'KASHKAI' -> 'QASHQAI'."""
    model = normalize_model(raw)
    return _MODELS_BY_SOUND.get(_sound_key(model), model)


def normalize_plate(raw: str) -> str:
    compact = "".join(raw.upper().split()).replace("-", "")
    return compact.translate(_PLATE_CYRILLIC_TO_LATIN)


def is_valid_plate(plate: str) -> bool:
    return bool(_PLATE_PATTERN.match(plate))


def normalize_vin(raw: str) -> str:
    return "".join(raw.upper().split())


def is_valid_vin(vin: str) -> bool:
    return bool(_VIN_PATTERN.match(vin))
