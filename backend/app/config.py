from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"


def load_project_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


@dataclass(frozen=True)
class AgentSettings:
    enabled: bool
    provider_label: str
    api_base_url: str
    api_key: str | None
    model: str
    temperature: float
    timeout_seconds: float


def get_agent_settings() -> AgentSettings:
    load_project_env()

    api_base_url = (
        _get_str("OPT_SIM_AGENT_API_BASE_URL")
        or _get_str("CODEX_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = _get_str("OPT_SIM_AGENT_API_KEY") or _get_str("CODEX_API_KEY")
    model = _get_str("OPT_SIM_AGENT_MODEL") or _get_str("CODEX_MODEL") or "gpt-4.1-mini"
    provider_label = _get_str("OPT_SIM_AGENT_PROVIDER_LABEL") or "OpenAI-compatible"

    return AgentSettings(
        enabled=_get_bool("OPT_SIM_AGENT_ENABLED", True),
        provider_label=provider_label,
        api_base_url=api_base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        temperature=_get_float("OPT_SIM_AGENT_TEMPERATURE", 0.2),
        timeout_seconds=_get_float("OPT_SIM_AGENT_TIMEOUT_SECONDS", 20.0),
    )
