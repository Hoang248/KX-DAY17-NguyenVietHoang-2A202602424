from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = (
    "openai",
    "custom",
    "gemini",
    "anthropic",
    "ollama",
    "openrouter",
)

_PROVIDER_ALIASES = {
    "openai": "openai",
    "open-ai": "openai",
    "open_ai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_generative_ai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",  # common typo retained for scaffold compatibility
    "claude": "anthropic",
    "ollama": "ollama",
    "local": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Configuration shared by the optional live-provider integrations."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a canonical provider name or fail with an actionable error."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("Provider name must be a non-empty string.")

    normalized = value.strip().lower().replace(" ", "-")
    try:
        return _PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(
            f"Unsupported provider {value!r}. Expected one of: {supported}."
        ) from exc


def _optional_dependency_error(provider: str, package: str, error: ImportError) -> RuntimeError:
    """Build a consistent error without importing optional SDKs at module load."""

    return RuntimeError(
        f"Provider {provider!r} requires optional package {package!r}. "
        "Install the provider dependency or run the agent in offline mode."
    )


def _common_kwargs(config: ProviderConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model": config.model_name,
        "temperature": config.temperature,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    return kwargs


def build_chat_model(config: ProviderConfig):
    """Instantiate the selected LangChain chat model lazily.

    Offline tests can import the project without any provider SDK installed. A
    missing SDK is reported only when the caller explicitly asks for a live
    model, and constructor errors are allowed to retain their provider detail.
    """

    provider = normalize_provider(config.provider)

    if provider in {"openai", "custom"}:
        if provider == "custom" and not config.base_url:
            raise ValueError("The custom provider requires ProviderConfig.base_url.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise _optional_dependency_error(provider, "langchain-openai", exc) from exc
        return ChatOpenAI(**_common_kwargs(config))

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise _optional_dependency_error(provider, "langchain-google-genai", exc) from exc
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
        }
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise _optional_dependency_error(provider, "langchain-anthropic", exc) from exc
        return ChatAnthropic(**_common_kwargs(config))

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise _optional_dependency_error(provider, "langchain-ollama", exc) from exc
        return ChatOllama(**_common_kwargs(config))

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError as exc:
            raise _optional_dependency_error(provider, "langchain-openrouter", exc) from exc
        return ChatOpenRouter(**_common_kwargs(config))

    # normalize_provider currently makes this unreachable, but keeping the
    # guard makes future additions fail closed rather than silently misroute.
    raise ValueError(f"Unsupported provider {provider!r}.")
