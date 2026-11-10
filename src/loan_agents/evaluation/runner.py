"""Running the cases through the service, with the model or with the case file's answers."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Awaitable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from loan_agents import __version__
from loan_agents.adapters.llm_langchain import LangChainExtractor
from loan_agents.contracts import (
    CertificateReading,
    Channel,
    LoanRequest,
    ManagerDecision,
    RequestExtraction,
    TurnResult,
)
from loan_agents.core import intake
from loan_agents.core.policy import Policy
from loan_agents.evaluation.cases import Certificate, EvalCase, Photo, load_cases, photo_sample
from loan_agents.evaluation.report import case_report, summary
from loan_agents.evaluation.scoring import (
    CaseResult,
    check_data,
    check_decision,
    check_manager,
    check_no_early_decision,
    check_photo_reply,
    score_message,
    score_reading,
)
from loan_agents.fakes import FakeExtractor, InMemoryCrm, RecordingMessenger
from loan_agents.ports import Extractor
from loan_agents.samples import Sample, render_sample
from loan_agents.service import LoanIntakeService, TurnFailed
from loan_agents.settings import ConfigError, Settings, load_policy

# --- extractors -------------------------------------------------------------------------------


class AnswerKeyExtractor:
    """Plays a perfect model: answers messages from the case file and photos from their key."""

    def __init__(self, cases: Iterable[EvalCase]) -> None:
        self._answers: dict[str, RequestExtraction] = {}
        for case in cases:
            for step in case.steps:
                if step.text is None:
                    continue
                answer = step.expect.as_extraction()
                if self._answers.setdefault(_text_key(step.text), answer) != answer:
                    raise ConfigError(f"case {case.id}: {step.text!r} has another answer elsewhere")
        self._photos = FakeExtractor()

    async def extract_request(self, text: str) -> RequestExtraction:
        return self._answers.get(_text_key(text), RequestExtraction())

    async def read_certificate(self, image: bytes, mime_type: str) -> CertificateReading:
        return await self._photos.read_certificate(image, mime_type)


def _text_key(text: str) -> str:
    return " ".join(text.split())


@dataclass
class TokenCounter:
    input: int = 0
    output: int = 0

    def add(self, task: str, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input += input_tokens or 0
        self.output += output_tokens or 0


@dataclass
class ModelAnswer:
    kind: Literal["message", "photo"]
    answer: RequestExtraction | CertificateReading


class RecordingExtractor:
    """Passes every call through and keeps the answer, so it can be scored against the case."""

    def __init__(self, inner: Extractor) -> None:
        self._inner = inner
        self.answers: list[ModelAnswer] = []

    async def extract_request(self, text: str) -> RequestExtraction:
        answer = await self._inner.extract_request(text)
        self.answers.append(ModelAnswer("message", answer))
        return answer

    async def read_certificate(self, image: bytes, mime_type: str) -> CertificateReading:
        answer = await self._inner.read_certificate(image, mime_type)
        self.answers.append(ModelAnswer("photo", answer))
        return answer


# --- one case ------------------------------------------------------------------------------------


@dataclass
class _Run:
    service: LoanIntakeService
    extractor: RecordingExtractor
    tokens: TokenCounter
    messenger: RecordingMessenger
    crm: InMemoryCrm
    policy: Policy
    today: date
    run_id: str


async def _run_case(run: _Run, case: EvalCase) -> CaseResult:
    result = CaseResult(case.id, case.about, f"web-eval-{run.run_id}-{case.id}")
    tokens_before = (run.tokens.input, run.tokens.output)
    try:
        await _play(run, case, result)
    except TurnFailed as error:
        result.error = str(error)
    result.input_tokens = run.tokens.input - tokens_before[0]
    result.output_tokens = run.tokens.output - tokens_before[1]
    return result


async def _timed(result: CaseResult, kind: str, turn: Awaitable[TurnResult]) -> TurnResult:
    started = time.monotonic()
    outcome = await turn
    result.turns.append((kind, round((time.monotonic() - started) * 1000, 1)))
    return outcome


async def _play(run: _Run, case: EvalCase, result: CaseResult) -> None:
    case_id = result.session_id
    service = run.service
    told = list((await _timed(result, "start", service.start(case_id, Channel.WEB))).replies)
    expected_request = LoanRequest()
    accepted: Certificate | None = None
    for number, step in enumerate(case.steps, start=1):
        label = f"step {number}"
        answers_before = len(run.extractor.answers)
        sample: Sample | None = None
        if step.say is not None:
            turn = service.handle_text(case_id, Channel.WEB, step.say)
            replies = (await _timed(result, "message", turn)).replies
        elif step.send is not None:
            sample = photo_sample(case.id, number, step.send)
            turn = service.handle_document(
                case_id, Channel.WEB, render_sample(sample), "image/png", step.caption
            )
            replies = (await _timed(result, "photo", turn)).replies
        told += replies
        answers = run.extractor.answers[answers_before:]

        if step.text is not None:
            expected_request = intake.merge_extraction(
                expected_request, step.expect.as_extraction(), run.policy, run.today.year
            ).request
            message = next((a.answer for a in answers if a.kind == "message"), None)
            if isinstance(message, RequestExtraction):
                result.fields.add(score_message(label, step.expect, message))
            else:
                result.problems.append(f"{label}: the message never reached the model")
        if sample is not None:
            truth = CertificateReading.model_validate(sample.answer_key)
            reading = next((a.answer for a in answers if a.kind == "photo"), None)
            if isinstance(reading, CertificateReading):
                result.reading.add(score_reading(label, truth, reading))
            else:
                result.problems.append(f"{label}: the photo never reached the model")
            check_photo_reply(result, label, step, replies)
            if step.rejected is None and isinstance(step.send, Photo):
                accepted = step.send.certificate

    cards = [card for card in run.messenger.approval_requests if card.case_id == case_id]
    contract = cards[0].contract_text if cards else None
    view = await service.view(case_id)
    # A case stopped short shows its missing data in the field notes already.
    if check_manager(result, case.outcome, len(cards), contract, view):
        check_data(result, expected_request, accepted if cards else None, view)

    if cards and case.decision is not None:
        decision = ManagerDecision(action=case.decision, manager="eval", comment=case.comment)
        decided = await _timed(result, "decision", service.decide(case_id, decision))
        check_decision(result, decision, decided, run.crm.deals.get(case_id))
    check_no_early_decision(result, told, case.comment)


# --- the run ------------------------------------------------------------------------------------


@dataclass
class EvalRun:
    run_id: str
    model: str
    today: date
    results: list[CaseResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)


async def run_eval(
    settings: Settings, cases_file: Path, *, fake: bool, only: Sequence[str]
) -> EvalRun:
    """Runs the cases and prints a line per case, then the summary.

    Raises ExtractorError when the model does not answer a first call made to load it.
    """
    from loan_agents.wiring import build_runtime

    case_file = load_cases(cases_file)
    unknown = sorted(set(only) - {case.id for case in case_file.cases})
    if unknown:
        raise ConfigError(f"{cases_file}: no cases {unknown}")
    cases = [case for case in case_file.cases if not only or case.id in only]
    policy = load_policy(settings.config_dir / "policy.yaml")
    tokens = TokenCounter()
    inner: Extractor
    if fake:
        inner, model = AnswerKeyExtractor(case_file.cases), "the case file's answers (--fake)"
    else:
        inner = LangChainExtractor(
            settings.llm_model,
            ollama_base_url=settings.ollama_base_url,
            timeout_s=settings.llm_timeout_s,
            on_usage=tokens.add,
        )
        model = settings.llm_model
        # Loads the model, so that the first case is not timed with the load.
        await inner.extract_request("Добрий день")
        tokens.input = tokens.output = 0

    run = EvalRun(time.strftime("%y%m%d-%H%M%S"), model, case_file.today, [])
    print(f"loan-intake-agents {__version__} eval-cases: {len(cases)} cases, model {model}\n")
    recorder = RecordingExtractor(inner)
    messenger, crm = RecordingMessenger(), InMemoryCrm()
    with tempfile.TemporaryDirectory(prefix="loan-agents-eval-") as uploads:
        # Cases stay in memory and photos in a temporary folder, away from the real ones.
        isolated = settings.model_copy(update={"database_url": None, "data_dir": Path(uploads)})
        async with build_runtime(
            isolated,
            fake=fake,
            extractor=recorder,
            messenger=messenger,
            crm=crm,
            today=lambda: case_file.today,
        ) as runtime:
            state = _Run(
                runtime.service, recorder, tokens, messenger, crm, policy, run.today, run.run_id
            )
            for case in cases:
                result = await _run_case(state, case)
                run.results.append(result)
                print(case_report(result))

    print("\n" + summary(run.results))
    if settings.langfuse_public_key is not None:
        print(
            f"Langfuse: one session per case, named web-eval-{run.run_id}-<case>, "
            f"at {settings.langfuse_base_url}"
        )
    return run
