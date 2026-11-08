from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from loan_agents.core.policy import Policy
from loan_agents.samples import SAMPLES, render_sample
from loan_agents.settings import PROJECT_ROOT, Settings, load_policy

TODAY = date(2026, 11, 8)
MANAGER_TOKEN = "test-manager-token"


@pytest.fixture(autouse=True)
def _ignore_developer_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values exported in the developer's shell must not leak into the tests."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def policy() -> Policy:
    return load_policy(PROJECT_ROOT / "config" / "policy.yaml")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        manager_api_token=MANAGER_TOKEN,
        telegram_bot_token=None,
        database_url=None,
        b24_webhook_url=None,
    )


def sample_bytes(file_name: str) -> bytes:
    for sample in SAMPLES:
        if sample.file_name == file_name:
            return render_sample(sample)
    raise KeyError(file_name)
