"""The whole agent team on fakes, driven through the service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from loan_agents.contracts import (
    CaseStatus,
    Channel,
    DealStage,
    DecisionAction,
    DocumentIssue,
    ManagerDecision,
    QuestionTopic,
    RequestExtraction,
)
from loan_agents.core import texts_uk as texts
from loan_agents.fakes import FakeExtractor, InMemoryCrm, RecordingMessenger
from loan_agents.ports import ExtractorError
from loan_agents.service import CaseNotFound, DecisionNotPending, InputError, TurnFailed
from loan_agents.settings import Settings
from loan_agents.wiring import Runtime, build_runtime
from tests.conftest import TODAY, sample_bytes

CASE = "web-test"
ANSWERS = (
    "Мене звати Тарас Шевченко",
    "067 123 45 67",
    "200 тис грн на 12 місяців",
    "Шкода Октавія 2018 року",
)


@asynccontextmanager
async def running(settings: Settings, **overrides: object) -> AsyncIterator[Runtime]:
    async with build_runtime(settings, fake=True, today=lambda: TODAY, **overrides) as runtime:
        yield runtime


async def answer_all(runtime: Runtime, answers: tuple[str, ...] = ANSWERS) -> None:
    await runtime.service.start(CASE, Channel.WEB)
    for text in answers:
        await runtime.service.handle_text(CASE, Channel.WEB, text)


def approve(action: DecisionAction = DecisionAction.APPROVE, comment: str | None = None):
    return ManagerDecision(action=action, manager="test", comment=comment)


async def test_happy_path_reaches_manager_and_closes(settings: Settings) -> None:
    messenger, crm = RecordingMessenger(), InMemoryCrm()
    async with running(settings, messenger=messenger, crm=crm) as runtime:
        service = runtime.service
        first = await service.start(CASE, Channel.WEB)
        assert first.replies[0] == texts.GREETING
        assert first.replies[1] == texts.ask_field("full_name", runtime_policy(), TODAY.year)

        for text in ANSWERS:
            await service.handle_text(CASE, Channel.WEB, text)
        result = await service.handle_document(
            CASE, Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )

        assert result.awaiting_manager
        assert result.status is CaseStatus.AWAITING_MANAGER
        assert result.replies == [texts.DOCUMENT_ACCEPTED, texts.SUBMITTED]
        card = messenger.approval_requests[0].card
        assert "+38067***4567" in card
        assert "+380671234567" not in card
        assert "Позначки: немає" in card
        assert messenger.approval_requests[0].contract_text is not None
        deal = crm.deals[CASE]
        assert deal.stage is DealStage.REVIEW
        assert deal.deal.last_name == "Шевченко"
        assert deal.deal.amount_uah == 200_000
        assert any("чернетку договору" in comment for comment in deal.comments)

        decided = await service.decide(CASE, approve())
        assert decided.status is CaseStatus.APPROVED
        assert decided.replies == [texts.DECISION_APPROVED]
        assert crm.deals[CASE].stage is DealStage.APPROVED

        after = await service.handle_text(CASE, Channel.WEB, "Дякую!")
        assert after.replies == [texts.CASE_CLOSED]


def runtime_policy():
    from loan_agents.settings import PROJECT_ROOT, load_policy

    return load_policy(PROJECT_ROOT / "config" / "policy.yaml")


async def test_rate_question_is_answered_from_approved_text(settings: Settings) -> None:
    async with running(settings) as runtime:
        await runtime.service.start(CASE, Channel.WEB)
        result = await runtime.service.handle_text(CASE, Channel.WEB, "Яка у вас ставка?")
        assert result.replies[0] == texts.faq_answer(QuestionTopic.RATE, runtime_policy())


async def test_out_of_range_amount_is_explained_and_asked_again(settings: Settings) -> None:
    async with running(settings) as runtime:
        await answer_all(runtime, ANSWERS[:2])
        result = await runtime.service.handle_text(CASE, Channel.WEB, "5 тис грн")
        policy = runtime_policy()
        assert result.replies == [
            texts.invalid_field("amount_uah", policy, TODAY.year),
            texts.ask_field("amount_uah", policy, TODAY.year),
        ]


async def test_client_messages_wait_while_manager_decides(settings: Settings) -> None:
    crm = InMemoryCrm()
    async with running(settings, crm=crm) as runtime:
        await answer_all(runtime)
        await runtime.service.handle_document(
            CASE, Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )
        calls_before = list(crm.calls)
        waiting = await runtime.service.handle_text(CASE, Channel.WEB, "Ну що там?")
        assert waiting.replies == [texts.WAITING_FOR_MANAGER]
        assert crm.calls == calls_before  # the graph did not re-run its CRM steps


async def test_need_documents_loops_back_to_same_deal(settings: Settings) -> None:
    messenger, crm = RecordingMessenger(), InMemoryCrm()
    async with running(settings, messenger=messenger, crm=crm) as runtime:
        await answer_all(runtime)
        photo = sample_bytes("certificate-ok.png")
        await runtime.service.handle_document(CASE, Channel.WEB, photo, "image/png")

        asked = await runtime.service.decide(
            CASE, approve(DecisionAction.NEED_DOCUMENTS, "Фото розмите")
        )
        assert asked.status is CaseStatus.COLLECTING
        assert asked.replies == [texts.decision_need_documents("Фото розмите")]
        assert crm.deals[CASE].stage is DealStage.NEW

        again = await runtime.service.handle_document(CASE, Channel.WEB, photo, "image/png")
        assert again.awaiting_manager
        assert len(crm.deals) == 1
        assert len(messenger.approval_requests) == 2


async def test_reject_moves_deal_to_rejected(settings: Settings) -> None:
    crm = InMemoryCrm()
    async with running(settings, crm=crm) as runtime:
        await answer_all(runtime)
        await runtime.service.handle_document(
            CASE, Channel.WEB, sample_bytes("certificate-ok.png"), "image/png"
        )
        result = await runtime.service.decide(CASE, approve(DecisionAction.REJECT))
        assert result.status is CaseStatus.REJECTED
        assert crm.deals[CASE].stage is DealStage.REJECTED


async def test_wrong_document_is_explained_once(settings: Settings) -> None:
    async with running(settings) as runtime:
        await answer_all(runtime)
        result = await runtime.service.handle_document(
            CASE, Channel.WEB, sample_bytes("not-a-certificate.png"), "image/png"
        )
        assert result.replies == [texts.photo_issue([DocumentIssue.NOT_A_CERTIFICATE])]
        assert result.status is CaseStatus.COLLECTING
        reminder = await runtime.service.handle_text(CASE, Channel.WEB, "Зараз знайду")
        assert reminder.replies == [texts.ASK_DOCUMENT]


async def test_someone_elses_car_is_flagged_and_gets_no_contract(settings: Settings) -> None:
    messenger = RecordingMessenger()
    async with running(settings, messenger=messenger) as runtime:
        await answer_all(runtime)
        result = await runtime.service.handle_document(
            CASE, Channel.WEB, sample_bytes("certificate-other-owner.png"), "image/png"
        )
        assert result.awaiting_manager
        request = messenger.approval_requests[0]
        assert request.contract_text is None
        assert "власник за документом — не заявник" in request.card
        assert "марка в документі інша" in request.card
        assert "Договір: не сформовано через позначки" in request.card


async def test_old_car_without_price_is_flagged(settings: Settings) -> None:
    messenger = RecordingMessenger()
    async with running(settings, messenger=messenger) as runtime:
        await answer_all(runtime)
        await runtime.service.handle_document(
            CASE, Channel.WEB, sample_bytes("certificate-old-car.png"), "image/png"
        )
        card = messenger.approval_requests[0].card
        assert "авто старше за ліміт (20 р.)" in card
        assert "немає ринкової ціни" in card


async def test_decision_without_pending_case_is_refused(settings: Settings) -> None:
    async with running(settings) as runtime:
        await runtime.service.start(CASE, Channel.WEB)
        with pytest.raises(DecisionNotPending):
            await runtime.service.decide(CASE, approve())


async def test_reset_forgets_case(settings: Settings) -> None:
    async with running(settings) as runtime:
        await runtime.service.start(CASE, Channel.WEB)
        await runtime.service.reset(CASE)
        with pytest.raises(CaseNotFound):
            await runtime.service.view(CASE)


@pytest.mark.parametrize(
    ("case_id", "data", "mime_type"),
    [
        ("bad id", b"x", "image/png"),
        (CASE, b"", "image/png"),
        (CASE, b"not an image", "image/png"),
        (CASE, b"%PDF-1.7", "application/pdf"),
        (CASE, b"0" * (11 * 1024 * 1024), "image/jpeg"),
    ],
)
async def test_bad_uploads_are_refused(
    settings: Settings, case_id: str, data: bytes, mime_type: str
) -> None:
    async with running(settings) as runtime:
        with pytest.raises(InputError):
            await runtime.service.handle_document(case_id, Channel.WEB, data, mime_type)


async def test_empty_text_is_refused(settings: Settings) -> None:
    async with running(settings) as runtime:
        with pytest.raises(InputError):
            await runtime.service.handle_text(CASE, Channel.WEB, "   ")


class _BrokenExtractor(FakeExtractor):
    async def extract_request(self, text: str) -> RequestExtraction:
        raise ExtractorError("model offline")


async def test_failing_model_is_reported_to_client(settings: Settings) -> None:
    messenger = RecordingMessenger()
    async with running(settings, messenger=messenger, extractor=_BrokenExtractor()) as runtime:
        await runtime.service.start(CASE, Channel.WEB)
        with pytest.raises(TurnFailed, match="model offline"):
            await runtime.service.handle_text(CASE, Channel.WEB, "Мене звати Тарас Шевченко")
        assert messenger.client_messages[-1] == (CASE, texts.TECHNICAL_PROBLEM)


async def test_parallel_messages_for_one_case_run_sequentially(settings: Settings) -> None:
    async with running(settings) as runtime:
        await runtime.service.start(CASE, Channel.WEB)
        await asyncio.gather(
            runtime.service.handle_text(CASE, Channel.WEB, "Мене звати Тарас Шевченко"),
            runtime.service.handle_text(CASE, Channel.WEB, "067 123 45 67"),
        )
        view = await runtime.service.view(CASE)
        assert view.request["full_name"] == "Т. Ш."
        assert view.request["phone"] == "+38067***4567"
