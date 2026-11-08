"""The car price table from config/market_prices.csv, validated when loaded."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from loan_agents.core.vehicles import normalize_make, normalize_model

_COLUMNS = ("make", "model", "year_from", "year_to", "price_uah")


class PriceTableError(ValueError):
    pass


@dataclass(frozen=True)
class _Row:
    year_from: int
    year_to: int
    price_uah: int


class CsvPriceBook:
    def __init__(self, rows: dict[tuple[str, str], list[_Row]]) -> None:
        self._rows = rows

    @classmethod
    def load(cls, path: Path) -> CsvPriceBook:
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != _COLUMNS:
                    raise PriceTableError(f"{path}: columns must be {', '.join(_COLUMNS)}")
                rows: dict[tuple[str, str], list[_Row]] = {}
                for line_number, raw in enumerate(reader, start=2):
                    key, row = _parse_row(path, line_number, raw)
                    for other in rows.get(key, []):
                        if row.year_from <= other.year_to and other.year_from <= row.year_to:
                            raise PriceTableError(
                                f"{path}:{line_number}: years overlap another {key[0]} {key[1]} row"
                            )
                    rows.setdefault(key, []).append(row)
        except FileNotFoundError as error:
            raise PriceTableError(f"{path}: not found") from error
        if not rows:
            raise PriceTableError(f"{path}: no prices")
        return cls(rows)

    def find_price_uah(self, make: str, model: str, year: int) -> int | None:
        for row in self._rows.get((normalize_make(make), normalize_model(model)), []):
            if row.year_from <= year <= row.year_to:
                return row.price_uah
        return None


def _parse_row(path: Path, line_number: int, raw: dict[str, str]) -> tuple[tuple[str, str], _Row]:
    try:
        row = _Row(int(raw["year_from"]), int(raw["year_to"]), int(raw["price_uah"]))
    except (TypeError, ValueError) as error:
        raise PriceTableError(f"{path}:{line_number}: years and price must be integers") from error
    make, model = normalize_make(raw["make"] or ""), normalize_model(raw["model"] or "")
    if not make or not model:
        raise PriceTableError(f"{path}:{line_number}: make and model are required")
    if row.year_from > row.year_to:
        raise PriceTableError(f"{path}:{line_number}: year_from is after year_to")
    if row.price_uah <= 0:
        raise PriceTableError(f"{path}:{line_number}: price must be positive")
    return (make, model), row
