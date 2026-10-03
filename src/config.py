from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, memory thresholds and optional model configurations."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_local_dotenv(root: Path) -> None:
    """Load a local ``.env`` when available without making it a hard dependency."""

    dotenv_path = root / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return

    if not dotenv_path.is_file():
        return

    # Small fallback parser for KEY=VALUE lines. It deliberately ignores
    # exports, interpolation and malformed lines instead of guessing secrets.
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number.") from exc


def _env_positive_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be greater than zero.")
    return value


def _provider_api_key(provider: str, judge: bool = False) -> str | None:
    prefix = "JUDGE_" if judge else ""
    specific = {
        "openai": (f"{prefix}OPENAI_API_KEY", "OPENAI_API_KEY"),
        "custom": (f"{prefix}CUSTOM_API_KEY", "CUSTOM_API_KEY", "OPENAI_API_KEY"),
        "gemini": (f"{prefix}GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "anthropic": (f"{prefix}ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        "ollama": (f"{prefix}OLLAMA_API_KEY", "OLLAMA_API_KEY"),
        "openrouter": (f"{prefix}OPENROUTER_API_KEY", "OPENROUTER_API_KEY"),
    }[provider]
    names = ((f"{prefix}API_KEY",) if judge else ()) + specific
    return _env(*names)


def _provider_base_url(provider: str, judge: bool = False) -> str | None:
    prefix = "JUDGE_" if judge else ""
    names = {
        "custom": (f"{prefix}CUSTOM_BASE_URL", "CUSTOM_BASE_URL"),
        "ollama": (f"{prefix}OLLAMA_BASE_URL", "OLLAMA_BASE_URL"),
        "openrouter": (f"{prefix}OPENROUTER_BASE_URL", "OPENROUTER_BASE_URL"),
    }.get(provider, ())
    return _env(*names)


def _make_provider_config(
    provider: str,
    model_name: str,
    temperature: float,
    *,
    judge: bool = False,
) -> ProviderConfig:
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=_provider_api_key(provider, judge=judge),
        base_url=_provider_base_url(provider, judge=judge),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load deterministic local defaults and optional live-provider settings."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    _load_local_dotenv(root)

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    provider = normalize_provider(_env("LLM_PROVIDER", default="openai") or "openai")
    judge_provider = normalize_provider(
        _env("JUDGE_PROVIDER", "LLM_JUDGE_PROVIDER", default=provider) or provider
    )

    model = _make_provider_config(
        provider,
        _env("LLM_MODEL", default="gpt-4o-mini") or "gpt-4o-mini",
        _env_float("LLM_TEMPERATURE", 0.0),
    )
    judge_model = _make_provider_config(
        judge_provider,
        _env("JUDGE_MODEL", "LLM_JUDGE_MODEL", default="gpt-4o-mini") or "gpt-4o-mini",
        _env_float("JUDGE_TEMPERATURE", 0.0),
        judge=True,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=_env_positive_int("COMPACT_THRESHOLD_TOKENS", 1200),
        compact_keep_messages=_env_positive_int("COMPACT_KEEP_MESSAGES", 6),
        model=model,
        judge_model=judge_model,
    )
