"""eval-cases: the case file, the scoring, and whole runs with the answer key as the model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from loan_agents.cli import main
from loan_agents.contracts import (
    CaseStatus,
    CaseView,
    CertificateReading,
    CrmDeal,
    DealStage,
    DecisionAction,
    DocumentIssue,
    ManagerDecision,
    RequestExtraction,
    TurnResult,
    Valuation,
)
from loan_agents.core import texts_uk as texts
from loan_agents.evaluation import CASES_FILE, load_cases, run_eval
from loan_agents.evaluation.cases import Expect, Outcome
from loan_agents.evaluation.runner import AnswerKeyExtractor
from loan_agents.evaluation.scoring import (
    CaseResult,
    check_decision,
    check_manager,
    check_no_early_decision,
    score_message,
    score_reading,
)
from loan_agents.fakes import FakeExtractor, InMemoryCrm
from loan_agents.fakes.crm import FakeDeal
from loan_agents.samples import Sample, render_sample
from loan_agents.settings import ConfigError, Settings


@pytest.fixture
def cli_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> Settings:
    monkeypatch.setattr("loan_agents.cli.load_settings", lambda: settings)
    return settings


def test_twenty_cases_five_of_them_with_bad_or_missing_documents() -> None:
    cases = load_cases(CASES_FILE).cases
    bad_documents = [
        case.id
        for case in cases
        if not any(step.send for step in case.steps)
        or any(step.rejected for step in case.steps)
        or case.outcome.document_flags
    ]
    assert len(cases) == 20
    assert bad_documents == ["e16", "e17", "e18", "e19", "e20"]


async def test_every_case_passes_when_model_makes_no_mistakes(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    run = await run_eval(settings, CASES_FILE, fake=True, only=[])
    failed = [result.id for result in run.results if not result.passed]
    assert failed == [], capsys.readouterr().out
    assert "Cases passed: 20/20" in capsys.readouterr().out


async def test_misheard_amount_fails_case_even_though_it_reaches_manager(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = AnswerKeyExtractor.extract_request

    async def misheard(self: AnswerKeyExtractor, text: str) -> RequestExtraction:
        answer = await original(self, text)
        if answer.amount_uah == 200_000:
            return answer.model_copy(update={"amount_uah": 20_000})
        return answer

    monkeypatch.setattr(AnswerKeyExtractor, "extract_request", misheard)
    run = await run_eval(settings, CASES_FILE, fake=True, only=["e02"])
    (result,) = run.results
    assert not result.passed
    assert "step 4: amount_uah 20000, expected 200000" in result.fields.notes
    assert "the case holds amount_uah 20000, expected 200000" in result.problems
    assert result.breaches == []


async def test_model_name_written_as_it_sounds_still_matches_certificate(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What qwen3-vl wrote for "Ніссан Кашкай 2017".
    original = AnswerKeyExtractor.extract_request

    async def by_sound(self: AnswerKeyExtractor, text: str) -> RequestExtraction:
        answer = await original(self, text)
        if answer.car_model == "QASHQAI":
            return answer.model_copy(update={"car_model": "KASHKAI"})
        return answer

    monkeypatch.setattr(AnswerKeyExtractor, "extract_request", by_sound)
    run = await run_eval(settings, CASES_FILE, fake=True, only=["e08"])
    (result,) = run.results
    assert result.passed, result.problems
    assert result.fields.wrong == 0


async def test_misread_model_is_flagged_for_manager_without_contract(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # What qwen3-vl does with the stamp across the fields.
    original = FakeExtractor.read_certificate

    async def misread(self: FakeExtractor, image: bytes, mime_type: str) -> CertificateReading:
        reading = await original(self, image, mime_type)
        return reading.model_copy(update={"model": "OSTAVIA"})

    monkeypatch.setattr(FakeExtractor, "read_certificate", misread)
    run = await run_eval(settings, CASES_FILE, fake=True, only=["e15"])
    (result,) = run.results
    assert "step 2: model 'OSTAVIA', expected 'OCTAVIA'" in result.reading.notes
    assert result.problems[:3] == [
        "document flags ['model_mismatch'], expected none",
        "policy flags ['no_market_price'], expected none",
        "no contract was drafted",
    ]
    assert result.breaches == []


async def test_accepted_photo_that_should_be_rejected_fails_case(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = FakeExtractor.read_certificate
    case = next(case for case in load_cases(CASES_FILE).cases if case.id == "e17")
    certificate = case.steps[2].send
    assert certificate is not None and certificate != "receipt"

    async def fooled(self: FakeExtractor, image: bytes, mime_type: str) -> CertificateReading:
        reading = await original(self, image, mime_type)
        if reading.is_vehicle_certificate:
            return reading
        printed = certificate.certificate.printed().model_dump()
        return CertificateReading(is_vehicle_certificate=True, **printed)

    monkeypatch.setattr(FakeExtractor, "read_certificate", fooled)
    run = await run_eval(settings, CASES_FILE, fake=True, only=["e17"])
    (result,) = run.results
    assert result.problems[0].startswith(
        "step 2: the photo should be turned down (not_a_certificate); the bot said: "
        + texts.DOCUMENT_ACCEPTED
    )


async def test_decision_missing_from_crm_fails_case(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def lost(self: InMemoryCrm, deal_id: str, stage: DealStage) -> None:
        return None

    monkeypatch.setattr(InMemoryCrm, "set_stage", lost)
    run = await run_eval(settings, CASES_FILE, fake=True, only=["e01"])
    (result,) = run.results
    assert result.problems == ["CRM stage review, expected approved"]


def test_message_is_scored_on_values_not_on_how_they_are_written() -> None:
    expected = Expect(full_name="Мар'яна Гнатюк", phone="+380501000106", question="documents")
    got = RequestExtraction(full_name="марʼяна гнатюк", phone="050 100 01 06")
    score = score_message("step 1", expected, got)
    assert (score.right, score.missed, score.wrong, score.extra) == (2, 1, 0, 0)
    assert score.notes == ["step 1: question missed, expected 'documents'"]


def test_values_read_from_unreadable_photo_count_as_extra() -> None:
    truth = CertificateReading(is_vehicle_certificate=True)
    guess = CertificateReading(
        is_vehicle_certificate=True, plate_number="AA1234BB", make="SKODA", year=2018
    )
    score = score_reading("step 2", truth, guess)
    assert (score.right, score.extra, score.expected) == (1, 3, 1)


def _view(flags: list[str], certificate: dict[str, Any] | None) -> CaseView:
    valuation = Valuation(
        market_value_uah=900_000,
        max_loan_uah=540_000,
        requested_uah=220_000,
        term_months=24,
        loan_to_value=0.2444,
        annual_rate_pct=18.0,
        monthly_payment_uah=10_984,
        flags=[],
    )
    return CaseView(
        case_id="web-eval-x-e18",
        status=CaseStatus.AWAITING_MANAGER,
        awaiting_manager=True,
        request={},
        certificate=certificate,
        valuation=valuation,
        crm=None,
        flags=flags,
        transcript=[texts.SUBMITTED],
    )


def test_guardrails_catch_contract_for_flagged_case_and_card_without_certificate() -> None:
    result = CaseResult("e18", "about", "web-eval-x-e18")
    outcome = Outcome(manager=True, document_flags=[DocumentIssue.OWNER_MISMATCH])
    check_manager(result, outcome, 2, "draft", _view(["owner_mismatch"], None))
    assert result.breaches == [
        "the manager got 2 cards for one case",
        "reached the manager without a certificate",
        "a contract was drafted for a flagged case",
    ]


def test_guardrails_catch_decision_sent_before_manager_decided() -> None:
    result = CaseResult("e01", "about", "web-eval-x-e01")
    check_no_early_decision(result, [texts.GREETING, texts.DECISION_APPROVED], None)
    assert result.breaches == ["the client was told a decision before the manager made one"]


def test_decision_must_reach_status_crm_and_client() -> None:
    result = CaseResult("e08", "about", "web-eval-x-e08")
    decision = ManagerDecision(action=DecisionAction.REJECT, manager="eval")
    decided = TurnResult(
        case_id="web-eval-x-e08",
        status=CaseStatus.APPROVED,
        replies=[texts.DECISION_APPROVED],
        awaiting_manager=False,
    )
    deal = FakeDeal(
        deal=CrmDeal(
            case_id="web-eval-x-e08",
            title="deal",
            first_name="Юлія",
            phone="+380971000108",
            amount_uah=250_000,
            stage=DealStage.REVIEW,
            comment="",
        ),
        contact_id="1",
        deal_id="2",
        stage=DealStage.APPROVED,
    )
    check_decision(result, decision, decided, deal)
    assert result.problems == [
        "status approved after reject, expected rejected",
        "CRM stage approved, expected rejected",
        "the client was not told the decision; the bot said: " + texts.DECISION_APPROVED,
    ]


VALID_CASE: dict[str, Any] = {
    "id": "x01",
    "about": "a valid case to break",
    "steps": [
        {
            "say": "Олена Коваленко, 067 100 01 01",
            "expect": {"full_name": "Олена Коваленко", "phone": "+380671000101"},
        },
        {
            "send": {
                "certificate": {
                    "plate_number": "KA1001AB",
                    "vin": "JTNB11HK103000101",
                    "make": "TOYOTA",
                    "model": "CAMRY",
                    "year": 2019,
                    "owner_full_name": "КОВАЛЕНКО ОЛЕНА ПЕТРІВНА",
                }
            }
        },
    ],
    "outcome": {"manager": True, "contract": True},
    "decision": "approve",
}


def _write_cases(path: Path, *cases: dict[str, Any]) -> Path:
    path.write_text(
        yaml.safe_dump({"today": "2026-11-09", "cases": list(cases)}, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _changed(**changes: Any) -> dict[str, Any]:
    case = json.loads(json.dumps(VALID_CASE))
    for dotted, value in changes.items():
        *parents, last = dotted.split("__")
        target = case
        for key in parents:
            target = target[int(key)] if key.isdigit() else target[key]
        target[last] = value
    return case


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (_changed(steps__1__send__certificate__vin="JTNB11HK1O3000101"), "VIN"),
        (_changed(steps__1__send__certificate__plate_number="KA101AB"), "plate"),
        (_changed(steps__0__expect__phone="067 100 01 01"), "'\\+380671000101'"),
        (_changed(steps__1__send="receipt"), "receipt must be rejected"),
        (_changed(steps__0__rejected="unreadable"), "caption and rejected go with send"),
        (
            _changed(outcome={"manager": True, "contract": True, "policy_flags": ["car_too_old"]}),
            "nothing is flagged",
        ),
        (_changed(outcome={"manager": False}), "needs a decision"),
        (_changed(colour="green"), "Extra inputs are not permitted"),
    ],
    ids=[
        "vin",
        "plate",
        "phone-form",
        "receipt",
        "rejected-text",
        "contract-flagged",
        "decision",
        "unknown-key",
    ],
)
def test_case_file_errors_stop_run_and_show_location(
    tmp_path: Path, case: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        load_cases(_write_cases(tmp_path / "cases.yaml", case))


def test_case_ids_are_unique(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="used more than once"):
        load_cases(_write_cases(tmp_path / "cases.yaml", VALID_CASE, VALID_CASE))


def test_one_message_cannot_have_two_answers(tmp_path: Path) -> None:
    other = _changed(id="x02", steps__0__expect={"full_name": "Олена Коваленко"})
    cases = load_cases(_write_cases(tmp_path / "cases.yaml", VALID_CASE, other)).cases
    with pytest.raises(ConfigError, match="another answer"):
        AnswerKeyExtractor(cases)


def test_blurred_photo_has_empty_answer_key() -> None:
    photo = load_cases(CASES_FILE).cases[0].steps[1].send
    assert photo is not None and photo != "receipt"
    printed = {"is_vehicle_certificate": True} | photo.certificate.printed().model_dump()
    blurred = Sample("blurred.png", printed, blur_px=10)
    assert blurred.answer_key == {"is_vehicle_certificate": True}
    assert Sample("sharp.png", printed).answer_key == printed
    assert render_sample(blurred) != render_sample(Sample("sharp.png", printed))


def test_eval_cases_writes_report_and_exits_0_when_all_pass(
    cli_settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval-cases", "--fake", "--only", "e01,e17", "--out", str(tmp_path)]) == 0
    (report_file,) = tmp_path.glob("cases-*.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert [case["id"] for case in report["cases"]] == ["e01", "e17"]
    assert all(case["passed"] for case in report["cases"])
    assert "Cases passed: 2/2" in capsys.readouterr().out


def test_eval_cases_names_unknown_case_ids(
    cli_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["eval-cases", "--fake", "--only", "e99"]) == 2
    assert "no cases ['e99']" in capsys.readouterr().err


def test_eval_cases_stops_when_model_does_not_answer(
    cli_settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    cli_settings.ollama_base_url = "http://127.0.0.1:9"
    assert main(["eval-cases", "--only", "e01"]) == 1
    assert "The model did not answer" in capsys.readouterr().out
