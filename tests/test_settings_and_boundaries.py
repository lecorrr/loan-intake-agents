from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from loan_agents.settings import PROJECT_ROOT, ConfigError, Settings, load_policy
from loan_agents.wiring import build_runtime

SRC = PROJECT_ROOT / "src" / "loan_agents"


def test_webhook_url_is_normalised() -> None:
    settings = Settings(_env_file=None, b24_webhook_url="https://firm.bitrix24.ua/rest/1/abc123")
    assert settings.b24_webhook_url == "https://firm.bitrix24.ua/rest/1/abc123/"


@pytest.mark.parametrize(
    "url", ["https://firm.bitrix24.ua/rest/abc/", "ftp://x/rest/1/a/", "firm.bitrix24.ua"]
)
def test_malformed_webhook_urls_fail_at_startup(url: str) -> None:
    with pytest.raises(ValidationError, match="b24_webhook_url"):
        Settings(_env_file=None, b24_webhook_url=url)


def test_a_bot_needs_a_manager_chat() -> None:
    with pytest.raises(ValidationError, match="MANAGER_CHAT_ID"):
        Settings(_env_file=None, telegram_bot_token="123:abc", manager_chat_id=None)


def test_model_spec_needs_a_provider() -> None:
    with pytest.raises(ValidationError, match="provider:model"):
        Settings(_env_file=None, llm_model="qwen3-vl")


def test_the_example_env_file_is_valid() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert settings.telegram_bot_token is None
    assert settings.manager_chat_id is None
    assert settings.b24_webhook_url is None
    assert settings.database_url is None


def test_empty_values_mean_not_set(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=\nMANAGER_CHAT_ID=\nLLM_TIMEOUT_S=\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.telegram_bot_token is None
    assert settings.manager_chat_id is None
    assert settings.llm_timeout_s == 180


def test_unknown_keys_in_the_env_file_are_errors(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODLE=ollama:qwen3-vl:8b\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="llm_modle"):
        Settings(_env_file=env_file)


def test_describe_hides_secrets_and_the_webhook_code() -> None:
    settings = Settings(
        _env_file=None,
        b24_webhook_url="https://firm.bitrix24.ua/rest/1/abc123/",
        telegram_bot_token="123:secret-token",
        manager_chat_id=5,
    )
    shown = json.dumps(settings.describe(), default=str)
    assert "abc123" not in shown
    assert "secret-token" not in shown
    assert "/rest/1/***/" in shown


@pytest.mark.parametrize(
    ("file_name", "content", "message"),
    [
        ("market_prices.csv", "make,model\nSKODA,OCTAVIA\n", "market_prices.csv: columns"),
        ("policy.yaml", "max_loan_to_value: [", "policy.yaml: not valid YAML"),
    ],
)
async def test_broken_config_files_stop_startup_with_their_name(
    settings: Settings, tmp_path: Path, file_name: str, content: str, message: str
) -> None:
    config_dir = tmp_path / "config"
    shutil.copytree(PROJECT_ROOT / "config", config_dir)
    (config_dir / file_name).write_text(content, encoding="utf-8")
    broken = settings.model_copy(update={"config_dir": config_dir})
    with pytest.raises(ConfigError, match=message):
        async with build_runtime(broken, fake=True):
            pass


def test_a_missing_config_file_names_the_project_folder(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="LOAN_AGENTS_HOME"):
        load_policy(tmp_path / "policy.yaml")


def test_core_loads_without_io_libraries_or_outer_modules() -> None:
    forbidden = [
        "httpx",
        "aiogram",
        "psycopg",
        "langchain",
        "langchain_core",
        "langgraph",
        "fastapi",
        "loan_agents.adapters",
        "loan_agents.agents",
        "loan_agents.service",
        "loan_agents.fakes",
        "loan_agents.wiring",
    ]
    modules = sorted(path.stem for path in (SRC / "core").glob("*.py") if path.stem != "__init__")
    script = (
        "import sys\n"
        + "".join(f"import loan_agents.core.{name}\n" for name in modules)
        + f"print([m for m in {forbidden!r} if m in sys.modules])"
    )
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout
    assert output.strip() == "[]"


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted((SRC / "agents").glob("*.py")), ids=lambda p: p.name)
def test_agents_use_ports_not_adapters(path: Path) -> None:
    bad = {
        name
        for name in _imports(path)
        if name.startswith(("loan_agents.adapters", "loan_agents.fakes", "loan_agents.wiring"))
    }
    assert bad == set()


@pytest.mark.parametrize("path", sorted((SRC / "adapters").glob("*.py")), ids=lambda p: p.name)
def test_adapters_stay_independent(path: Path) -> None:
    bad = {
        name
        for name in _imports(path)
        if name.startswith(("loan_agents.adapters", "loan_agents.agents", "loan_agents.service"))
    }
    assert bad == set()
