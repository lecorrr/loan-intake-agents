"""Every text the client, the manager and the CRM see, in Ukrainian, kept in one place."""

from __future__ import annotations

from collections.abc import Sequence

from loan_agents.contracts import (
    DecisionAction,
    DocumentIssue,
    LoanRequest,
    ManagerDecision,
    PolicyFlag,
    QuestionTopic,
    Valuation,
    VehicleCertificate,
)
from loan_agents.core.masking import mask_phone, mask_plate, mask_vin
from loan_agents.core.policy import Policy
from loan_agents.core.vehicles import MIN_CAR_YEAR

NBSP = " "


def format_uah(amount: int) -> str:
    return f"{amount:,}".replace(",", NBSP) + f"{NBSP}грн"


def format_pct(share: float) -> str:
    return f"{share * 100:.0f}%"


# --- client conversation --------------------------------------------------------------------

GREETING = (
    "Вітаю! Я допоможу подати заявку на кредит під заставу авто: поставлю кілька запитань "
    "і попрошу фото техпаспорта. Рішення завжди ухвалює менеджер."
)

_ASK = {
    "full_name": "Як вас звати? Напишіть, будь ласка, прізвище та ім'я.",
    "phone": "Ваш номер телефону для зв'язку? Наприклад: +380 67 123 45 67.",
    "amount_uah": "Яку суму ви хочете отримати? Від {amount_min} до {amount_max}.",
    "term_months": "На який строк, у місяцях? Від {term_min} до {term_max}.",
    "car_make": "Яка марка авто, що буде заставою? Наприклад: Skoda.",
    "car_model": "Яка модель авто? Наприклад: Octavia.",
    "car_year": "Якого року випуску авто?",
    "region": "У якій області ви перебуваєте?",
}

_INVALID = {
    "full_name": "Не вдалося прийняти ім'я: напишіть, будь ласка, прізвище та ім'я повністю.",
    "phone": (
        "Не вдалося розпізнати номер: потрібен український номер, наприклад +380 67 123 45 67."
    ),
    "amount_uah": "Не вдалося прийняти суму: можливо від {amount_min} до {amount_max}.",
    "term_months": "Не вдалося прийняти строк: можливо від {term_min} до {term_max} місяців.",
    "car_make": "Не вдалося розпізнати марку авто.",
    "car_model": "Не вдалося розпізнати модель авто.",
    "car_year": "Не вдалося прийняти рік випуску: можливо від {min_year} до {current_year}.",
    "region": "Не вдалося розпізнати область.",
}


def _limits(policy: Policy, current_year: int) -> dict[str, str | int]:
    return {
        "amount_min": format_uah(policy.amount_uah.min),
        "amount_max": format_uah(policy.amount_uah.max),
        "term_min": policy.term_months.min,
        "term_max": policy.term_months.max,
        "min_year": MIN_CAR_YEAR,
        "current_year": current_year,
    }


def ask_field(field: str, policy: Policy, current_year: int) -> str:
    return _ASK[field].format(**_limits(policy, current_year))


def invalid_field(field: str, policy: Policy, current_year: int) -> str:
    return _INVALID[field].format(**_limits(policy, current_year))


ASK_DOCUMENT = (
    "Дякую! Тепер надішліть, будь ласка, фото свідоцтва про реєстрацію транспортного засобу "
    "(техпаспорта) — бік із VIN-кодом і даними власника."
)

_PHOTO_ISSUE = {
    DocumentIssue.NOT_A_CERTIFICATE: (
        "Схоже, на фото не техпаспорт. Надішліть, будь ласка, фото свідоцтва про реєстрацію "
        "транспортного засобу."
    ),
    DocumentIssue.UNREADABLE: (
        "Не вдалося прочитати всі дані з фото. Сфотографуйте, будь ласка, техпаспорт ще раз: "
        "рівно, при доброму світлі, без відблисків."
    ),
    DocumentIssue.INVALID_VIN: (
        "VIN-код на фото прочитався з помилкою. Надішліть, будь ласка, чіткіше фото техпаспорта."
    ),
    DocumentIssue.INVALID_PLATE: (
        "Номерний знак на фото прочитався з помилкою. Надішліть, будь ласка, чіткіше фото "
        "техпаспорта."
    ),
}


def photo_issue(issues: Sequence[DocumentIssue]) -> str:
    return _PHOTO_ISSUE[issues[0]]


DOCUMENT_ACCEPTED = "Дякую, техпаспорт прочитано. Перевіряю дані й готую заявку для менеджера."
SUBMITTED = "Заявку передано менеджеру. Я напишу вам, щойно він ухвалить рішення."
WAITING_FOR_MANAGER = "Ваша заявка вже в менеджера. Я напишу, щойно буде рішення."
DECISION_APPROVED = (
    "Менеджер попередньо погодив вашу заявку. Він зв'яжеться з вами, щоб домовитися про "
    "огляд авто та підписання договору."
)
DECISION_REJECTED = (
    "На жаль, зараз ми не можемо запропонувати кредит за цією заявкою. Дякуємо, що звернулися."
)
CASE_CLOSED = "Цю заявку вже розглянуто. Щоб подати нову, надішліть /new."
UNSUPPORTED_MESSAGE = "Надішліть, будь ласка, текст або фото техпаспорта."
TECHNICAL_PROBLEM = (
    "Вибачте, сталася технічна помилка. Спробуйте, будь ласка, ще раз за кілька хвилин."
)
UPLOAD_NOT_IMAGE = "Надішліть, будь ласка, фото техпаспорта як зображення JPEG, PNG або WebP."
UPLOAD_TOO_LARGE = "Файл завеликий: надішліть, будь ласка, фото до {max_mb} МБ."
DOWNLOAD_FAILED = "Не вдалося отримати файл. Надішліть, будь ласка, фото ще раз."


def decision_need_documents(comment: str | None) -> str:
    text = "Менеджеру потрібне ще одне фото техпаспорта. Надішліть його, будь ласка, у цей чат."
    if comment:
        text += f"\nКоментар менеджера: {comment}"
    return text


def faq_answer(topic: QuestionTopic, policy: Policy) -> str:
    if topic is QuestionTopic.RATE:
        return (
            f"Ставка — від {policy.annual_rate_pct:g}% річних, у гривні. "
            "Остаточні умови підтверджує менеджер."
        )
    if topic is QuestionTopic.DOCUMENTS:
        return (
            "Потрібні паспорт або ID-картка, ідентифікаційний код і техпаспорт авто. "
            "Авто залишається у вас."
        )
    if topic is QuestionTopic.TIMING:
        return (
            "Після перевірки документів менеджер зв'язується з вами, щоб домовитися про огляд "
            "авто та підписання договору."
        )
    return "Передам ваше запитання менеджеру разом із заявкою."


# --- manager and CRM ------------------------------------------------------------------------

DECISION_LABEL = {
    DecisionAction.APPROVE: "погоджено",
    DecisionAction.REJECT: "відмовлено",
    DecisionAction.NEED_DOCUMENTS: "потрібні документи",
}

DECISION_BUTTON = {
    DecisionAction.APPROVE: "✅ Погодити",
    DecisionAction.NEED_DOCUMENTS: "📄 Потрібні документи",
    DecisionAction.REJECT: "❌ Відмовити",
}

MANAGER_ONLY = "Ця кнопка лише для менеджера."
UNKNOWN_ACTION = "Невідома дія."
SAVING_DECISION = "Зберігаю рішення…"


def manager_already_decided(case_id: str) -> str:
    return f"Заявка {case_id}: рішення вже ухвалено."


def manager_decision_failed(case_id: str) -> str:
    return f"Заявка {case_id}: не вдалося зберегти рішення, подробиці в лозі сервісу."


def decided_card(card: str, action: DecisionAction) -> str:
    return f"{card}\n\nРішення: {DECISION_LABEL[action]}"


def flag_texts(
    policy_flags: Sequence[PolicyFlag],
    document_flags: Sequence[DocumentIssue],
    request: LoanRequest,
    policy: Policy,
) -> list[str]:
    texts = []
    for flag in policy_flags:
        if flag is PolicyFlag.CAR_TOO_OLD:
            texts.append(f"авто старше за ліміт ({policy.max_car_age_years} р.)")
        elif flag is PolicyFlag.NO_MARKET_PRICE:
            texts.append("немає ринкової ціни в довіднику, потрібна оцінка вручну")
        elif flag is PolicyFlag.ABOVE_MAX_LOAN_TO_VALUE:
            texts.append("сума вища за ліміт LTV")
    for issue in document_flags:
        if issue is DocumentIssue.MAKE_MISMATCH:
            texts.append(f"марка в документі інша, ніж у заявці ({request.car_make})")
        elif issue is DocumentIssue.MODEL_MISMATCH:
            texts.append(f"модель у документі інша, ніж у заявці ({request.car_model})")
        elif issue is DocumentIssue.YEAR_MISMATCH:
            texts.append(f"рік у документі інший, ніж у заявці ({request.car_year})")
        elif issue is DocumentIssue.OWNER_MISMATCH:
            texts.append("власник за документом — не заявник")
    return texts


def _valuation_lines(valuation: Valuation, policy: Policy) -> list[str]:
    if valuation.market_value_uah is None or valuation.loan_to_value is None:
        price = "Ринкова ціна: немає в довіднику"
    else:
        price = (
            f"Ринкова ціна: {format_uah(valuation.market_value_uah)} · "
            f"LTV {format_pct(valuation.loan_to_value)} "
            f"(ліміт {format_pct(policy.max_loan_to_value)}, "
            f"до {format_uah(valuation.max_loan_uah or 0)})"
        )
    payment = (
        f"Платіж: ~{format_uah(valuation.monthly_payment_uah)}/міс. "
        f"при {valuation.annual_rate_pct:g}% річних"
    )
    return [price, payment]


def manager_card(
    *,
    case_id: str,
    request: LoanRequest,
    certificate: VehicleCertificate,
    valuation: Valuation,
    document_flags: Sequence[DocumentIssue],
    notes: Sequence[str],
    crm_deal_id: str,
    has_contract: bool,
    policy: Policy,
) -> str:
    """The Telegram card. Phone, plate and VIN are masked; the CRM holds the full values."""
    phone = mask_phone(request.phone) if request.phone else "—"
    flags = flag_texts(valuation.flags, document_flags, request, policy)
    lines = [
        f"Нова заявка {case_id}",
        f"Клієнт: {request.full_name or '—'}, {phone}",
        (
            f"Авто (за документом): {certificate.make} {certificate.model}, {certificate.year}; "
            f"номер {mask_plate(certificate.plate_number)}; VIN {mask_vin(certificate.vin)}"
        ),
        f"Сума: {format_uah(valuation.requested_uah)} на {valuation.term_months} міс.",
        *_valuation_lines(valuation, policy),
        "Позначки: " + ("; ".join(flags) if flags else "немає"),
    ]
    if notes:
        lines.append("Запитання клієнта: " + " | ".join(notes))
    lines.append(
        "Договір: чернетка в додатку" if has_contract else "Договір: не сформовано через позначки"
    )
    lines.append(f"CRM: угода #{crm_deal_id}")
    return "\n".join(lines)


def crm_deal_title(certificate: VehicleCertificate, amount_uah: int) -> str:
    return (
        f"Кредит під заставу: {certificate.make} {certificate.model} {certificate.year}, "
        f"{format_uah(amount_uah)}"
    )


def crm_summary(
    *,
    request: LoanRequest,
    certificate: VehicleCertificate,
    valuation: Valuation,
    document_flags: Sequence[DocumentIssue],
    notes: Sequence[str],
    policy: Policy,
) -> str:
    """The deal's comment in the CRM: full values, since the CRM is the system of record."""
    flags = flag_texts(valuation.flags, document_flags, request, policy)
    lines = [
        "Заявку зібрали агенти, рішення за менеджером.",
        f"Клієнт: {request.full_name}, {request.phone}, регіон: {request.region or '—'}",
        (
            f"Авто: {certificate.make} {certificate.model}, {certificate.year}, "
            f"номер {certificate.plate_number}, VIN {certificate.vin}"
        ),
        f"Власник за документом: {certificate.owner_full_name}",
        f"Сума: {format_uah(valuation.requested_uah)} на {valuation.term_months} міс.",
        *_valuation_lines(valuation, policy),
        "Позначки: " + ("; ".join(flags) if flags else "немає"),
    ]
    if notes:
        lines.append("Запитання клієнта: " + " | ".join(notes))
    return "\n".join(lines)


def crm_contract_comment(contract_text: str) -> str:
    return "Агенти підготували чернетку договору:\n\n" + contract_text


def crm_decision_comment(decision: ManagerDecision) -> str:
    text = f"Рішення менеджера ({decision.manager}): {DECISION_LABEL[decision.action]}"
    if decision.comment:
        text += f". Коментар: {decision.comment}"
    return text
