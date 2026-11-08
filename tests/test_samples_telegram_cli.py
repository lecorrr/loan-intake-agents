from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from loan_agents.adapters.telegram import (
    TelegramMessenger,
    case_id_for_chat,
    chat_id_for_case,
    decision_callback,
    parse_decision_callback,
)
from loan_agents.cli import MIN_OLLAMA_FOR_QWEN3_VL, _setup_logging, _version_tuple, main
from loan_agents.contracts import ApprovalRequest, CertificateReading, DecisionAction, LoanRequest
from loan_agents.core.documents import check_certificate
from loan_agents.fakes import FakeExtractor
from loan_agents.ports import ExtractorError
from loan_agents.samples import SAMPLES, read_truth, render_sample, write_samples
from loan_agents.settings import Settings


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda sample: sample.file_name)
async def test_samples_carry_valid_ground_truth(sample) -> None:
    data = render_sample(sample)
    assert data.startswith(b"\x89PNG") and len(data) > 10_000
    reading = await FakeExtractor().read_certificate(data, "image/png")
    assert reading == CertificateReading.model_validate(sample.truth)
    if reading.is_vehicle_certificate:
        check = check_certificate(reading, LoanRequest(), 2026)
        assert check.certificate is not None, "a certificate sample must pass validation itself"


def _changed_area(sample) -> tuple[int, int, int, int] | None:
    with_stamp = Image.open(io.BytesIO(render_sample(sample)))
    without = Image.open(io.BytesIO(render_sample(sample, stamp=False)))
    return ImageChops.difference(with_stamp.convert("RGB"), without.convert("RGB")).getbbox()


# Every field value sits left of x=680 or above y=450 (the layout in samples.py).
CLEAR_OF_FIELDS = (680, 450)


@pytest.mark.parametrize(
    "sample", [s for s in SAMPLES if s.stamp == "corner"], ids=lambda s: s.file_name
)
def test_the_regular_stamp_stays_clear_of_every_field(sample) -> None:
    left, top, _, _ = _changed_area(sample)
    assert left >= CLEAR_OF_FIELDS[0] and top >= CLEAR_OF_FIELDS[1]


def test_the_hard_sample_stamp_crosses_the_model_field() -> None:
    hard = next(s for s in SAMPLES if s.stamp == "across")
    left, top, right, bottom = _changed_area(hard)
    model_value = (570, 295, 720, 330)  # where "D.3 Модель" prints its value
    assert left < model_value[0] and top < model_value[1]
    assert right > model_value[2] and bottom > model_value[3]


def test_write_samples_and_read_truth_back(tmp_path: Path) -> None:
    paths = write_samples(tmp_path)
    assert len(paths) == len(SAMPLES)
    assert read_truth(paths[0]).vin == "TMBJJ7NE8J0123456"


@pytest.mark.parametrize("chat_id", [42, -1001234567890])
def test_case_and_chat_ids_round_trip(chat_id: int) -> None:
    assert chat_id_for_case(case_id_for_chat(chat_id)) == chat_id


def test_web_cases_have_no_chat() -> None:
    assert chat_id_for_case("web-abc") is None


@pytest.mark.parametrize("action", list(DecisionAction))
def test_decision_callbacks_round_trip(action: DecisionAction) -> None:
    data = decision_callback("tg--1001234567890", action)
    assert parse_decision_callback(data) == ("tg--1001234567890", action)


@pytest.mark.parametrize("data", ["", "dec:tg-1:maybe", "other:tg-1:approve", "dec::approve"])
def test_foreign_callbacks_are_ignored(data: str) -> None:
    assert parse_decision_callback(data) is None


def test_overlong_callback_data_is_refused() -> None:
    with pytest.raises(ValueError, match="64 bytes"):
        decision_callback("web-" + "x" * 64, DecisionAction.APPROVE)


class _StubBot:
    def __init__(self) -> None:
        self.messages: list[tuple[int, str]] = []
        self.documents: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str, **kwargs: object) -> None:
        self.messages.append((chat_id, text))

    async def send_document(self, chat_id: int, document, **kwargs: object) -> None:
        self.documents.append((chat_id, document.filename))


async def test_messenger_sends_client_replies_only_to_telegram_cases() -> None:
    bot = _StubBot()
    messenger = TelegramMessenger(bot, manager_chat_id=7)  # type: ignore[arg-type]
    await messenger.send_to_client("web-abc", ["ignored"])
    await messenger.send_to_client("tg-42", ["Привіт"])
    assert bot.messages == [(42, "Привіт")]


async def test_approval_goes_to_the_manager_with_the_draft() -> None:
    bot = _StubBot()
    messenger = TelegramMessenger(bot, manager_chat_id=7)  # type: ignore[arg-type]
    await messenger.send_approval_request(
        ApprovalRequest(case_id="tg-42", card="Нова заявка", contract_text="# Договір")
    )
    assert bot.messages == [(7, "Нова заявка")]
    assert bot.documents == [(7, "contract-draft-tg-42.md")]


@pytest.fixture
def cli_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> Settings:
    """The CLI reads the project's .env; tests use the isolated settings instead."""
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: settings)
    return settings


def test_demo_runs_end_to_end_on_fakes(
    cli_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["demo", "--fake"]) == 0
    output = capsys.readouterr().out
    assert "Картка для менеджера" in output
    assert "Проєкт договору" in output
    assert "Status: approved" in output


def test_eval_documents_scores_the_fake_reader(
    cli_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval-documents", "--fake"]) == 0
    assert re.search(r"Field accuracy: (\d+)/\1 = 100%", capsys.readouterr().out)


def test_eval_documents_shows_what_was_read(
    cli_settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = FakeExtractor.read_certificate

    async def misread(self: FakeExtractor, data: bytes, mime_type: str) -> CertificateReading:
        reading = await original(self, data, mime_type)
        if reading.model == "OCTAVIA":
            return reading.model_copy(update={"model": "OSTAVIA"})
        return reading

    monkeypatch.setattr(FakeExtractor, "read_certificate", misread)
    assert main(["eval-documents", "--fake", "--runs", "2"]) == 0
    output = capsys.readouterr().out
    assert "certificate-ok.png #2" in output
    assert "model read 'OSTAVIA', expected 'OCTAVIA'" in output
    assert not re.search(r"Field accuracy: (\d+)/\1 ", output)


@pytest.mark.parametrize("runs", ["0", "-1", "two"])
def test_eval_documents_needs_a_positive_run_count(cli_settings: Settings, runs: str) -> None:
    with pytest.raises(SystemExit):
        main(["eval-documents", "--fake", "--runs", runs])


def test_eval_documents_fails_when_the_model_fails(
    cli_settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(self: FakeExtractor, data: bytes, mime_type: str) -> CertificateReading:
        raise ExtractorError("model timed out")

    monkeypatch.setattr(FakeExtractor, "read_certificate", broken)
    assert main(["eval-documents", "--fake"]) == 1
    assert "model timed out" in capsys.readouterr().out


def test_demo_reports_a_failed_model_call(
    cli_settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def broken(self: FakeExtractor, *args: object, **kwargs: object) -> None:
        raise ExtractorError("connection refused")

    monkeypatch.setattr(FakeExtractor, "extract_request", broken)
    assert main(["demo", "--fake"]) == 1
    assert "A turn failed" in capsys.readouterr().out


@pytest.mark.parametrize(("text", "expected"), [("0.12.7", (0, 12, 7)), ("0.13.0-rc1", (0, 13, 0))])
def test_ollama_versions_are_compared_as_numbers(text: str, expected: tuple[int, int, int]) -> None:
    assert _version_tuple(text) == expected
    assert _version_tuple("0.9.10") < MIN_OLLAMA_FOR_QWEN3_VL <= _version_tuple("0.12.10")


def test_an_unreadable_ollama_version_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="cannot read"):
        _version_tuple("dev")


def test_request_urls_stay_out_of_the_logs_even_at_debug() -> None:
    _setup_logging("DEBUG")
    assert logging.getLogger("httpx").getEffectiveLevel() >= logging.WARNING


def test_make_samples_writes_files(cli_settings: Settings, tmp_path: Path) -> None:
    assert main(["make-samples", "--out", str(tmp_path)]) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(s.file_name for s in SAMPLES)
