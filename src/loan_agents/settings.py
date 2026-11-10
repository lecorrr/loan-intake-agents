"""Settings from the environment and .env, plus loading the policy and the contract template.

Unknown keys in .env are errors, so a misspelt setting fails at startup instead of being ignored.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from loan_agents.contracts import DealStage
from loan_agents.core.policy import Policy

PROJECT_ROOT = Path(os.environ.get("LOAN_AGENTS_HOME") or Path(__file__).resolve().parents[2])
ENV_FILE = PROJECT_ROOT / ".env"

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_SITE_URL = re.compile(r"^https?://[^/\s?#]+(/[^\s?#]*)?$")
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        # `TELEGRAM_BOT_TOKEN=` with nothing after it means "not set", not an empty token.
        env_ignore_empty=True,
        extra="forbid",
    )

    llm_model: str = "ollama:qwen3-vl:8b-instruct"
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_timeout_s: float = Field(180.0, gt=0, le=1800)

    # The address EspoCRM opens at in a browser. Empty: deals stay in memory.
    espocrm_url: str | None = None
    # The API user's key; `loan-agents crm-setup` creates the user and prints it.
    espocrm_api_key: SecretStr | None = None
    espocrm_stage_new: str = Field("Prospecting", min_length=1)
    espocrm_stage_review: str = Field("Qualification", min_length=1)
    espocrm_stage_approved: str = Field("Negotiation", min_length=1)
    espocrm_stage_rejected: str = Field("Closed Lost", min_length=1)
    # Only for docker compose, which creates EspoCRM's administrator on the first start, and for
    # `loan-agents crm-setup`. The running app signs in with the API key.
    espocrm_admin_username: str = "admin"
    espocrm_admin_password: SecretStr | None = None

    # Langfuse tracing; without keys nothing is traced. docker compose runs Langfuse on this
    # address with these keys (`--profile tracing`).
    langfuse_base_url: str = "http://127.0.0.1:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    # Only for docker compose: Langfuse creates this user when it starts and the user is missing.
    langfuse_admin_email: str = "admin@loan-intake.local"
    langfuse_admin_password: SecretStr | None = None

    telegram_bot_token: SecretStr | None = None
    manager_chat_id: int | None = None
    manager_api_token: SecretStr | None = None

    database_url: SecretStr | None = None

    data_dir: Path = PROJECT_ROOT / "data"
    config_dir: Path = PROJECT_ROOT / "config"
    templates_dir: Path = PROJECT_ROOT / "templates"
    max_upload_mb: int = Field(10, ge=1, le=50)
    log_level: str = "INFO"

    @field_validator("llm_model")
    @classmethod
    def _provider_prefix(cls, value: str) -> str:
        provider, _, model = value.partition(":")
        if not provider or not model:
            raise ValueError("must look like provider:model, e.g. ollama:qwen3-vl:8b-instruct")
        return value

    @field_validator("espocrm_url")
    @classmethod
    def _site_url_shape(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        value = value.strip().rstrip("/")
        if not _SITE_URL.match(value):
            raise ValueError("must be the address EspoCRM opens at, e.g. http://127.0.0.1:8090")
        if value.endswith("/api/v1"):
            raise ValueError("leave out /api/v1; give the address EspoCRM opens at")
        return value

    @field_validator("langfuse_base_url")
    @classmethod
    def _langfuse_url_shape(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not _SITE_URL.match(value):
            raise ValueError("must be the address Langfuse opens at, e.g. http://127.0.0.1:3000")
        return value

    @field_validator("langfuse_admin_email")
    @classmethod
    def _email_shape(cls, value: str) -> str:
        if not _EMAIL.match(value):
            raise ValueError("must be an email address")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        if value.upper() not in _LOG_LEVELS:
            raise ValueError(f"must be one of {_LOG_LEVELS}")
        return value.upper()

    @model_validator(mode="after")
    def _manager_chat_with_bot(self) -> Settings:
        if self.telegram_bot_token is not None and self.manager_chat_id is None:
            raise ValueError("MANAGER_CHAT_ID is required when TELEGRAM_BOT_TOKEN is set")
        return self

    @model_validator(mode="after")
    def _langfuse_keys_in_pairs(self) -> Settings:
        if (self.langfuse_public_key is None) != (self.langfuse_secret_key is None):
            raise ValueError("set both LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or neither")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def espocrm_stages(self) -> dict[DealStage, str]:
        return {
            DealStage.NEW: self.espocrm_stage_new,
            DealStage.REVIEW: self.espocrm_stage_review,
            DealStage.APPROVED: self.espocrm_stage_approved,
            DealStage.REJECTED: self.espocrm_stage_rejected,
        }

    def describe(self) -> dict[str, Any]:
        """Every setting with secrets hidden, for the startup log and `loan-agents doctor`."""
        shown: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                shown[name] = "set" if value.get_secret_value() else "empty"
            else:
                shown[name] = str(value) if isinstance(value, Path) else value
        return shown


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        raise ConfigError(
            describe_validation_error("settings (.env / environment)", error)
        ) from error


def describe_validation_error(source: str, error: ValidationError) -> str:
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "(root)"
        problems.append(f"{location}: {item['msg']}")
    return f"{source}: " + "; ".join(problems)


def _not_found(path: Path) -> ConfigError:
    return ConfigError(
        f"{path}: not found (project folder: {PROJECT_ROOT}; set LOAN_AGENTS_HOME if that is wrong)"
    )


def load_policy(path: Path) -> Policy:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise _not_found(path) from error
    except yaml.YAMLError as error:
        raise ConfigError(f"{path}: not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    try:
        return Policy.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(describe_validation_error(str(path), error)) from error


def load_template(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise _not_found(path) from error
    if not text.strip():
        raise ConfigError(f"{path}: empty template")
    return text
