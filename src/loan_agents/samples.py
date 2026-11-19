"""Synthetic registration certificates: invented data, stamped ЗРАЗОК / SAMPLE, plain layout.

The answer key is embedded in the PNG (iTXt chunk) for the fake extractor and eval-documents.
certificate-stamp-over-fields.png is the hard case; the blurred one has nothing readable.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont, PngImagePlugin

from loan_agents.contracts import CertificateReading
from loan_agents.fakes.extractor import TRUTH_KEY

_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",  # Debian, Ubuntu
    "/usr/share/fonts/TTF",  # Arch
    "/usr/share/fonts/dejavu",  # Fedora
)
_SIZE_PX = (1100, 700)
_BACKGROUND = (238, 243, 236)
_INK = (25, 30, 35)
_LABEL = (95, 105, 110)
_STAMP_ACROSS = (200, 30, 30, 48)
_STAMP_CORNER = (200, 30, 30, 150)

Stamp = Literal["corner", "across"]


@dataclass(frozen=True)
class Sample:
    file_name: str
    truth: dict[str, Any]
    colour: str = "СІРИЙ"
    registered: str = "12.03.2019"
    stamp: Stamp = "corner"
    # Gaussian blur radius; at 10 px no field on the certificate can be read.
    blur_px: int = 0

    @property
    def answer_key(self) -> dict[str, Any]:
        """What a perfect reader reads from the image: everything printed, or nothing if blurred."""
        if self.blur_px and self.truth["is_vehicle_certificate"]:
            return {"is_vehicle_certificate": True}
        return self.truth


_OK_TRUTH = {
    "is_vehicle_certificate": True,
    "plate_number": "AA1234BB",
    "vin": "TMBJJ7NE8J0123456",
    "make": "SKODA",
    "model": "OCTAVIA",
    "year": 2018,
    "owner_full_name": "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ",
}

SAMPLES = (
    Sample("certificate-ok.png", _OK_TRUTH),
    Sample(
        "certificate-other-owner.png",
        {
            "is_vehicle_certificate": True,
            "plate_number": "KA7351HC",
            "vin": "JTNB23HK903456789",
            "make": "TOYOTA",
            "model": "CAMRY",
            "year": 2019,
            "owner_full_name": "ОСТАПЕНКО ОКСАНА ІВАНІВНА",
        },
        colour="ЧОРНИЙ",
        registered="05.06.2020",
    ),
    Sample(
        "certificate-old-car.png",
        {
            "is_vehicle_certificate": True,
            "plate_number": "BC4821AK",
            "vin": "VF1LSRAB534567890",
            "make": "RENAULT",
            "model": "LOGAN",
            "year": 2004,
            "owner_full_name": "ШЕВЧЕНКО ТАРАС ГРИГОРОВИЧ",
        },
        colour="БІЛИЙ",
        registered="21.09.2011",
    ),
    Sample("not-a-certificate.png", {"is_vehicle_certificate": False}),
    Sample("certificate-stamp-over-fields.png", _OK_TRUTH, stamp="across"),
)


def _font_path(bold: bool) -> str:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for directory in _FONT_DIRS:
        path = Path(directory) / name
        if path.is_file():
            return str(path)
    raise RuntimeError(
        f"{name} not found in {', '.join(_FONT_DIRS)}; "
        "install fonts-dejavu-core (Ubuntu) or ttf-dejavu (Arch)"
    )


def _fonts() -> dict[str, ImageFont.FreeTypeFont]:
    regular, bold = _font_path(False), _font_path(True)
    return {
        "title": ImageFont.truetype(bold, 30),
        "label": ImageFont.truetype(regular, 20),
        "value": ImageFont.truetype(bold, 30),
        "note": ImageFont.truetype(regular, 18),
        "stamp_across": ImageFont.truetype(bold, 150),
        "stamp_corner": ImageFont.truetype(bold, 50),
    }


def _stamp_across(canvas: Image.Image, fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """A large, light diagonal ЗРАЗОК across the fields: the hard case."""
    font = fonts["stamp_across"]
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    left, top, right, bottom = draw.textbbox((0, 0), "ЗРАЗОК", font=font)
    x = (canvas.width - (right - left)) / 2 - left
    y = (canvas.height - (bottom - top)) / 2 - top
    draw.text((x, y), "ЗРАЗОК", font=font, fill=_STAMP_ACROSS)
    rotated = layer.rotate(18, resample=Image.Resampling.BICUBIC)
    return Image.alpha_composite(canvas, rotated)


def _stamp_corner(canvas: Image.Image, fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    """A small framed ЗРАЗОК in the empty bottom-right corner, clear of every field."""
    font = fonts["stamp_corner"]
    left, top, right, bottom = ImageDraw.Draw(canvas).textbbox((0, 0), "ЗРАЗОК", font=font)
    padding = 14
    width, height = right - left + 2 * padding, bottom - top + 2 * padding
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((2, 2, width - 3, height - 3), 10, outline=_STAMP_CORNER, width=4)
    draw.text((padding - left, padding - top), "ЗРАЗОК", font=font, fill=_STAMP_CORNER)
    rotated = layer.rotate(8, resample=Image.Resampling.BICUBIC, expand=True)
    position = (canvas.width - rotated.width - 50, canvas.height - rotated.height - 40)
    stamped = canvas.copy()
    stamped.alpha_composite(rotated, dest=position)
    return stamped


_STAMPS = {"corner": _stamp_corner, "across": _stamp_across}


def _draw_certificate(sample: Sample, fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    truth = sample.truth
    canvas = Image.new("RGBA", _SIZE_PX, (*_BACKGROUND, 255))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((12, 12, _SIZE_PX[0] - 12, _SIZE_PX[1] - 12), 28, outline=_INK, width=3)
    draw.text((50, 40), "СВІДОЦТВО ПРО РЕЄСТРАЦІЮ", font=fonts["title"], fill=_INK)
    draw.text((50, 80), "ТРАНСПОРТНОГО ЗАСОБУ (ЗРАЗОК)", font=fonts["title"], fill=_INK)
    # (label, value, x, y): the owner gets a full-width row because names run long.
    fields = [
        ("A. Номерний знак", truth["plate_number"], 50, 150),
        ("E. Ідентифікаційний номер (VIN)", truth["vin"], 570, 150),
        ("D.1 Марка", truth["make"], 50, 265),
        ("D.3 Модель", truth["model"], 570, 265),
        ("B. Рік випуску", str(truth["year"]), 50, 380),
        ("R. Колір", sample.colour, 330, 380),
        ("I. Дата реєстрації", sample.registered, 570, 380),
        ("C.1 Власник", truth["owner_full_name"], 50, 495),
    ]
    for label, value, x, y in fields:
        draw.text((x, y), label, font=fonts["label"], fill=_LABEL)
        draw.text((x, y + 30), value, font=fonts["value"], fill=_INK)
    draw.text(
        (50, _SIZE_PX[1] - 55),
        "Тестовий зразок з вигаданими даними. Не є документом.",
        font=fonts["note"],
        fill=_LABEL,
    )
    return canvas


def _draw_receipt(fonts: dict[str, ImageFont.FreeTypeFont]) -> Image.Image:
    canvas = Image.new("RGBA", _SIZE_PX, (250, 250, 247, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 40), "КВИТАНЦІЯ ПРО ОПЛАТУ (ЗРАЗОК)", font=fonts["title"], fill=_INK)
    lines = [
        ("Платник", "ПЕТРЕНКО ІВАН"),
        ("Призначення", "Комунальні послуги, жовтень"),
        ("Сума", "1 250,00 грн"),
        ("Дата", "03.10.2026"),
    ]
    for index, (label, value) in enumerate(lines):
        y = 130 + index * 110
        draw.text((50, y), label, font=fonts["label"], fill=_LABEL)
        draw.text((50, y + 30), value, font=fonts["value"], fill=_INK)
    return canvas


def render_sample(sample: Sample, *, stamp: bool = True) -> bytes:
    """The sample as PNG. stamp=False leaves the stamp off, to compare reads with and without."""
    fonts = _fonts()
    if sample.truth["is_vehicle_certificate"]:
        picture = _draw_certificate(sample, fonts)
    else:
        picture = _draw_receipt(fonts)
    if stamp:
        picture = _STAMPS[sample.stamp](picture, fonts)
    if sample.blur_px:
        picture = picture.filter(ImageFilter.GaussianBlur(sample.blur_px))
    info = PngImagePlugin.PngInfo()
    info.add_itxt(TRUTH_KEY, json.dumps(sample.answer_key, ensure_ascii=False))
    output = io.BytesIO()
    picture.convert("RGB").save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def write_samples(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for sample in SAMPLES:
        path = out_dir / sample.file_name
        path.write_bytes(render_sample(sample))
        written.append(path)
    return written


def read_truth(path: Path) -> CertificateReading:
    with Image.open(path) as picture:
        raw = picture.info.get(TRUTH_KEY)
    if raw is None:
        raise ValueError(f"{path}: no embedded ground truth ({TRUTH_KEY})")
    return CertificateReading.model_validate(json.loads(raw))
