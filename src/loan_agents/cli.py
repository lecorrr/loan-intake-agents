"""Command line: serve, demo, doctor, crm-setup, make-samples, eval-documents."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from loan_agents import __version__
from loan_agents.contracts import (
    ApprovalRequest,
    CertificateReading,
    Channel,
    DecisionAction,
    ManagerDecision,
)
from loan_agents.core.vehicles import (
    normalize_make,
    normalize_model,
    normalize_plate,
    normalize_vin,
)
from loan_agents.ports import ExtractorError
from loan_agents.settings import ENV_FILE, PROJECT_ROOT, ConfigError, Settings, load_settings

if TYPE_CHECKING:
    from loan_agents.service import LoanIntakeService

log = logging.getLogger("loan_agents.cli")

# qwen3-vl needs this Ollama release or newer; older ones fail to load the model.
MIN_OLLAMA_FOR_QWEN3_VL = (0, 12, 7)

DEMO_CASE_ID = "web-demo"
DEMO_MESSAGES = (
    "Добрий день! Мене звати Тарас Шевченко",
    "Мій номер 067 123 45 67",
    "Потрібно 200 тис грн на 12 місяців. Яка у вас ставка?",
    "Авто Шкода Октавія 2018 року",
)


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 1..65535, got {port}")
    return port


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {number}")
    return number


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"{path}: no such file")
    return path


def _existing_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"{path}: no such directory")
    return path


# httpx logs every request URL at INFO, and a CRM search by phone has the number in its URL.
QUIET_LOGGERS = ("httpx", "httpcore")


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# --- serve ----------------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    from loan_agents.api import create_app
    from loan_agents.wiring import check_crm_settings

    if not args.fake:
        # Here it stops with one line; inside the server it would stop with a traceback.
        check_crm_settings(settings)
    uvicorn.run(create_app(settings, fake=args.fake), host=args.host, port=args.port)
    return 0


# --- demo -----------------------------------------------------------------------------------


class _ConsoleMessenger:
    async def send_to_client(self, case_id: str, texts: list[str]) -> None:
        for text in texts:
            print(f"  БОТ: {text}")

    async def send_approval_request(self, request: ApprovalRequest) -> None:
        print("\n--- Картка для менеджера " + "-" * 50)
        print(request.card)
        if request.contract_text:
            print("\n--- Чернетка договору (додаток) " + "-" * 43)
            print(request.contract_text)
        print("-" * 75 + "\n")


async def _run_demo_case(service: LoanIntakeService, args: argparse.Namespace) -> int:
    await service.start(DEMO_CASE_ID, Channel.WEB)
    for text in DEMO_MESSAGES:
        print(f"\nКЛІЄНТ: {text}")
        await service.handle_text(DEMO_CASE_ID, Channel.WEB, text)
    print(f"\nКЛІЄНТ: [фото {args.document.name}]")
    result = await service.handle_document(
        DEMO_CASE_ID, Channel.WEB, args.document.read_bytes(), "image/png"
    )
    if not result.awaiting_manager:
        print("The case did not reach the manager; see the replies above.")
        return 1
    action = DecisionAction(args.decision)
    print(f"МЕНЕДЖЕР: {action.value}\n")
    await service.decide(
        DEMO_CASE_ID, ManagerDecision(action=action, manager="demo", comment=args.comment)
    )
    view = await service.view(DEMO_CASE_ID)
    print(f"\nStatus: {view.status.value}; CRM: {view.crm.model_dump() if view.crm else None}")
    return 0


async def _demo(args: argparse.Namespace, settings: Settings) -> int:
    from loan_agents.service import TurnFailed
    from loan_agents.wiring import build_runtime

    async with build_runtime(settings, fake=args.fake, messenger=_ConsoleMessenger()) as runtime:
        mode = "fake extractor, in-memory CRM" if args.fake else settings.llm_model
        print(f"loan-intake-agents {__version__} demo ({mode})\n")
        try:
            return await _run_demo_case(runtime.service, args)
        except TurnFailed:
            print(
                "\nA turn failed; the log above has the cause. `loan-agents doctor` checks setup."
            )
            return 1


def cmd_demo(args: argparse.Namespace, settings: Settings) -> int:
    return asyncio.run(_demo(args, settings))


# --- doctor ---------------------------------------------------------------------------------


def _version_tuple(text: str) -> tuple[int, int, int]:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        raise RuntimeError(f"cannot read the Ollama version from {text!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


class CheckWarning(Exception):
    """A doctor check that passed with a problem worth fixing; it does not fail the run."""


async def _check_ollama(settings: Settings) -> str:
    import httpx

    provider, _, model = settings.llm_model.partition(":")
    if provider != "ollama":
        return f"skipped: provider is {provider}"
    base_url = settings.ollama_base_url
    async with httpx.AsyncClient(timeout=10) as http:
        version = str((await http.get(f"{base_url}/api/version")).json()["version"])
        tags = (await http.get(f"{base_url}/api/tags")).json()
        if model.startswith("qwen3-vl") and _version_tuple(version) < MIN_OLLAMA_FOR_QWEN3_VL:
            needed = ".".join(str(part) for part in MIN_OLLAMA_FOR_QWEN3_VL)
            raise RuntimeError(f"Ollama {version} cannot run {model}; {needed} or newer is needed")
        names = {item["name"] for item in tags.get("models", [])}
        if model not in names and f"{model}:latest" not in names:
            raise RuntimeError(f"model {model} not pulled; run: ollama pull {model}")
        show = (await http.post(f"{base_url}/api/show", json={"model": model})).json()
    if "thinking" in (show.get("capabilities") or []):
        raise CheckWarning(
            f"Ollama {version} has {model}, a reasoning build. Some of these keep reasoning "
            "even when asked not to (qwen3-vl:8b does), which makes every call several times "
            "slower; output_tokens in the log shows it. An instruct build such as "
            "qwen3-vl:8b-instruct does not reason."
        )
    return f"Ollama {version} has {model}"


async def _check_espocrm(settings: Settings) -> str:
    from loan_agents.adapters.espocrm import EspoClient, EspoCrm, api_key_http

    if settings.espocrm_url is None:
        return "skipped: ESPOCRM_URL not set (deals go to an in-memory CRM)"
    if settings.espocrm_api_key is None:
        raise RuntimeError("ESPOCRM_API_KEY is empty; run: loan-agents crm-setup")
    api_key = settings.espocrm_api_key.get_secret_value()
    async with api_key_http(settings.espocrm_url, api_key) as http:
        crm = EspoCrm(
            EspoClient(http, max_attempts=1),
            settings.espocrm_stages,
            site_url=settings.espocrm_url,
        )
        return f"{settings.espocrm_url}: {await crm.check()}"


async def _check_postgres(settings: Settings) -> str:
    import psycopg

    if settings.database_url is None:
        return "skipped: DATABASE_URL not set (cases are lost on restart)"
    conninfo = settings.database_url.get_secret_value()
    async with await psycopg.AsyncConnection.connect(conninfo, connect_timeout=10) as connection:
        cursor = await connection.execute("SELECT version()")
        row = await cursor.fetchone()
    return str(row[0]).split(",")[0] if row else "connected"


async def _check_telegram(settings: Settings) -> str:
    from aiogram import Bot

    if settings.telegram_bot_token is None:
        return "skipped: TELEGRAM_BOT_TOKEN not set"
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    try:
        me = await bot.get_me()
        # "chat not found" here means the manager never pressed Start in a private chat with
        # the bot, or the bot is not a member of the manager group.
        chat = await bot.get_chat(settings.manager_chat_id)
    finally:
        await bot.session.close()
    return f"bot @{me.username}, manager chat {chat.id} ({chat.type})"


async def _doctor(settings: Settings) -> int:
    from loan_agents.adapters.prices_csv import CsvPriceBook
    from loan_agents.settings import load_policy, load_template

    print(f"loan-intake-agents {__version__}")
    for name, value in settings.describe().items():
        print(f"  {name} = {value}")
    checks = {
        "policy": lambda: _sync(load_policy, settings.config_dir / "policy.yaml"),
        "prices": lambda: _sync(CsvPriceBook.load, settings.config_dir / "market_prices.csv"),
        "template": lambda: _sync(
            load_template, settings.templates_dir / "loan_agreement_uk.md.j2"
        ),
        "ollama": lambda: _check_ollama(settings),
        "espocrm": lambda: _check_espocrm(settings),
        "postgres": lambda: _check_postgres(settings),
        "telegram": lambda: _check_telegram(settings),
    }
    failed = 0
    for name, check in checks.items():
        try:
            outcome = await check()
            print(f"OK    {name}: {outcome}")
        except CheckWarning as warning:
            print(f"WARN  {name}: {warning}")
        except Exception as error:  # every check reports its own failure and the rest still run
            failed += 1
            print(f"FAIL  {name}: {type(error).__name__}: {error}")
    return 1 if failed else 0


async def _sync(function: Callable[[Path], object], path: Path) -> str:
    function(path)
    return f"{path} loaded"


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    return asyncio.run(_doctor(settings))


# --- crm-setup ------------------------------------------------------------------------------


async def _crm_setup(settings: Settings, write_env: bool) -> int:
    from loan_agents.adapters.espocrm import EspoClient, EspoError, admin_http, set_up_crm
    from loan_agents.ports import CrmError

    if settings.espocrm_url is None:
        raise ConfigError("ESPOCRM_URL is not set; it is the address EspoCRM opens at")
    if settings.espocrm_admin_password is None:
        raise ConfigError("ESPOCRM_ADMIN_PASSWORD is not set")
    print(f"Setting up EspoCRM at {settings.espocrm_url} as {settings.espocrm_admin_username}")
    password = settings.espocrm_admin_password.get_secret_value()
    try:
        async with admin_http(
            settings.espocrm_url, settings.espocrm_admin_username, password
        ) as http:
            report = await set_up_crm(EspoClient(http))
    except CrmError as error:
        print(f"CRM setup failed: {error}", file=sys.stderr)
        status = error.status if isinstance(error, EspoError) else "not an HTTP error"
        if status in (None, 502, 503, 504):
            print(
                "EspoCRM may still be starting: its first start takes a minute or two, and "
                "`docker compose ps` shows it as healthy once it is ready.",
                file=sys.stderr,
            )
        elif status == 401:
            print(
                "Check ESPOCRM_ADMIN_USERNAME and ESPOCRM_ADMIN_PASSWORD. EspoCRM takes the "
                "password from .env only on its very first start; changing it there later "
                "does not change it in EspoCRM.",
                file=sys.stderr,
            )
        return 1
    for step in report.steps:
        print(f"  {step}")
    current = settings.espocrm_api_key
    if current is not None and current.get_secret_value() == report.api_key:
        print("ESPOCRM_API_KEY is already this user's key.")
    elif write_env:
        save_api_key(ENV_FILE, report.api_key)
        print(f"ESPOCRM_API_KEY saved in {ENV_FILE}")
    else:
        print("\nPut this line in .env (it replaces an empty ESPOCRM_API_KEY=):")
        print(f"ESPOCRM_API_KEY={report.api_key}")
    return 0


def save_api_key(env_file: Path, api_key: str) -> None:
    """Puts the key on the ESPOCRM_API_KEY line of .env; every other line stays as it is."""
    if not env_file.is_file():
        raise ConfigError(f"{env_file}: not found; put the ESPOCRM_API_KEY line in by hand")
    key_line = f"ESPOCRM_API_KEY={api_key}"
    lines: list[str] = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ESPOCRM_API_KEY="):
            lines.append(line)
        elif key_line not in lines:
            lines.append(key_line)
    if key_line not in lines:
        lines.append(key_line)
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_crm_setup(args: argparse.Namespace, settings: Settings) -> int:
    return asyncio.run(_crm_setup(settings, args.write_env))


# --- samples and eval -----------------------------------------------------------------------


def cmd_make_samples(args: argparse.Namespace, settings: Settings) -> int:
    from loan_agents.samples import write_samples

    for path in write_samples(args.out):
        print(f"wrote {path}")
    return 0


def _field_matches(name: str, truth: object, read: object) -> bool:
    if truth is None or read is None:
        return truth == read
    normalizers: dict[str, Callable[[str], str]] = {
        "plate_number": normalize_plate,
        "vin": normalize_vin,
        "make": normalize_make,
        "model": normalize_model,
        "owner_full_name": lambda value: " ".join(value.upper().split()),
    }
    if name in normalizers:
        return normalizers[name](str(truth)) == normalizers[name](str(read))
    return truth == read


async def _eval_documents(args: argparse.Namespace, settings: Settings) -> int:
    from loan_agents.adapters.llm_langchain import LangChainExtractor
    from loan_agents.fakes import FakeExtractor
    from loan_agents.samples import read_truth

    extractor = (
        FakeExtractor()
        if args.fake
        else LangChainExtractor(
            settings.llm_model,
            ollama_base_url=settings.ollama_base_url,
            timeout_s=settings.llm_timeout_s,
        )
    )
    paths = sorted(args.samples.glob("*.png"))
    if not paths:
        print(f"{args.samples}: no .png samples; run: loan-agents make-samples")
        return 1
    fields = list(CertificateReading.model_fields)
    correct = total = failed = 0
    for path in paths:
        truth = read_truth(path)
        image = path.read_bytes()
        for run in range(1, args.runs + 1):
            label = path.name if args.runs == 1 else f"{path.name} #{run}"
            total += len(fields)
            started = time.monotonic()
            try:
                reading = await extractor.read_certificate(image, "image/png")
            except ExtractorError as error:
                failed += 1
                print(f"{label:36} {'':>7}     error: {error}")
                continue
            elapsed_ms = (time.monotonic() - started) * 1000
            misses = [
                name
                for name in fields
                if not _field_matches(name, getattr(truth, name), getattr(reading, name))
            ]
            correct += len(fields) - len(misses)
            print(f"{label:36} {elapsed_ms:7.0f} ms  {_verdict(misses, truth, reading)}")
    print(f"\nField accuracy: {correct}/{total} = {correct / total:.0%}")
    if failed:
        print(f"{failed} of {len(paths) * args.runs} reads failed; the model call failed")
        return 1
    return 0


def _verdict(misses: list[str], truth: CertificateReading, reading: CertificateReading) -> str:
    if not misses:
        return "all fields right"
    # The samples are synthetic, so printing what was read is safe, and it is what a fix needs.
    return "wrong: " + "; ".join(
        f"{name} read {getattr(reading, name)!r}, expected {getattr(truth, name)!r}"
        for name in misses
    )


def cmd_eval_documents(args: argparse.Namespace, settings: Settings) -> int:
    return asyncio.run(_eval_documents(args, settings))


# --- entry point ----------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[argparse.Namespace, Settings], int]] = {
    "serve": cmd_serve,
    "demo": cmd_demo,
    "doctor": cmd_doctor,
    "make-samples": cmd_make_samples,
    "eval-documents": cmd_eval_documents,
    "crm-setup": cmd_crm_setup,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loan-agents", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the HTTP API and the Telegram bot")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=_port, default=8000)
    serve.add_argument("--fake", action="store_true", help="no model, in-memory CRM")

    demo = commands.add_parser("demo", help="run a scripted application in the terminal")
    demo.add_argument("--fake", action="store_true", help="no model, in-memory CRM")
    demo.add_argument(
        "--document",
        type=_existing_file,
        # A string default goes through `type`, so a missing sample fails at parse time.
        default=str(PROJECT_ROOT / "samples" / "certificate-ok.png"),
    )
    demo.add_argument(
        "--decision", choices=[action.value for action in DecisionAction], default="approve"
    )
    demo.add_argument("--comment", default=None)

    commands.add_parser("doctor", help="check settings, model, CRM, database and bot")

    samples = commands.add_parser("make-samples", help="write synthetic certificate images")
    samples.add_argument("--out", type=Path, default=PROJECT_ROOT / "samples")

    evaluate = commands.add_parser("eval-documents", help="score the document reader on samples")
    evaluate.add_argument("--samples", type=_existing_dir, default=str(PROJECT_ROOT / "samples"))
    evaluate.add_argument("--fake", action="store_true", help="score the fake extractor")
    evaluate.add_argument(
        "--runs", type=_positive_int, default=1, help="read every sample this many times"
    )

    crm_setup = commands.add_parser(
        "crm-setup", help="prepare EspoCRM once: currency, case field, role and API user"
    )
    crm_setup.add_argument(
        "--write-env", action="store_true", help="save the API key in .env instead of printing it"
    )

    missing = set(commands.choices) - set(COMMANDS)
    if missing:
        raise RuntimeError(f"commands without a handler: {sorted(missing)}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    _setup_logging(settings.log_level)
    try:
        return COMMANDS[args.command](args, settings)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
