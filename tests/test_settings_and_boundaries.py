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


@pytest.mark.parametrize("field", ["espocrm_url", "langfuse_base_url"])
@pytest.mark.parametrize("url", ["http://127.0.0.1:8090/", " https://crm.firm.ua/espo "])
def test_site_addresses_are_normalised(field: str, url: str) -> None:
    settings = Settings(_env_file=None, **{field: url})
    assert getattr(settings, field) == url.strip().rstrip("/")


@pytest.mark.parametrize(
    "url",
    [
        "127.0.0.1:8090",
        "ftp://crm.firm.ua",
        "http://127.0.0.1:8090/api/v1/",
        "http://127.0.0.1:8090/#Opportunity",
    ],
)
def test_malformed_crm_addresses_fail_at_startup(url: str) -> None:
    with pytest.raises(ValidationError, match="espocrm_url"):
        Settings(_env_file=None, espocrm_url=url)


@pytest.mark.parametrize(
    ("field", "value"),
    [("langfuse_base_url", "127.0.0.1:3000"), ("langfuse_admin_email", "admin")],
)
def test_malformed_langfuse_settings_fail_at_startup(field: str, value: str) -> None:
    with pytest.raises(ValidationError, match=field):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    "keys", [{"langfuse_public_key": "pk-lf-1"}, {"langfuse_secret_key": "sk-lf-1"}]
)
def test_langfuse_keys_come_in_pairs(keys: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"):
        Settings(_env_file=None, **keys)


def test_stage_names_cannot_be_empty() -> None:
    with pytest.raises(ValidationError, match="espocrm_stage_approved"):
        Settings(_env_file=None, espocrm_stage_approved="")


async def test_crm_address_without_api_key_stops_startup(settings: Settings) -> None:
    half_done = settings.model_copy(update={"espocrm_url": "http://127.0.0.1:8090"})
    with pytest.raises(ConfigError, match="crm-setup"):
        async with build_runtime(half_done):
            pass


def test_bot_needs_manager_chat() -> None:
    with pytest.raises(ValidationError, match="MANAGER_CHAT_ID"):
        Settings(_env_file=None, telegram_bot_token="123:abc", manager_chat_id=None)


def test_model_spec_needs_provider() -> None:
    with pytest.raises(ValidationError, match="provider:model"):
        Settings(_env_file=None, llm_model="qwen3-vl")


def test_example_env_file_is_valid() -> None:
    settings = Settings(_env_file=PROJECT_ROOT / ".env.example")
    assert settings.llm_model == Settings.model_fields["llm_model"].default
    assert settings.telegram_bot_token is None
    assert settings.manager_chat_id is None
    assert settings.espocrm_url is None
    assert settings.espocrm_api_key is None
    assert settings.langfuse_public_key is None
    assert settings.langfuse_secret_key is None
    assert settings.langfuse_admin_password is None
    assert settings.database_url is None


def test_empty_values_mean_not_set(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=\nMANAGER_CHAT_ID=\nLLM_TIMEOUT_S=\n", encoding="utf-8")
    settings = Settings(_env_file=env_file)
    assert settings.telegram_bot_token is None
    assert settings.manager_chat_id is None
    assert settings.llm_timeout_s == 180


def test_unknown_keys_in_env_file_are_errors(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODLE=ollama:qwen3-vl:8b\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="llm_modle"):
        Settings(_env_file=env_file)


def test_describe_hides_secrets() -> None:
    settings = Settings(
        _env_file=None,
        espocrm_url="http://127.0.0.1:8090",
        espocrm_api_key="api-key-123",
        espocrm_admin_password="admin-pass-456",
        telegram_bot_token="123:secret-token",
        manager_chat_id=5,
        langfuse_public_key="pk-lf-1",
        langfuse_secret_key="sk-lf-789",
        langfuse_admin_password="lf-pass-012",
    )
    shown = json.dumps(settings.describe(), default=str)
    for secret in ("api-key-123", "admin-pass-456", "secret-token", "sk-lf-789", "lf-pass-012"):
        assert secret not in shown
    assert settings.describe()["espocrm_api_key"] == "set"


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


def test_missing_config_file_names_project_folder(tmp_path: Path) -> None:
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
        "langfuse",
        "opentelemetry",
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
