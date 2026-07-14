from __future__ import annotations

from dataclasses import dataclass

from llm.gemini import GeminiError, generate_answer as gemini_generate_answer
from llm.gemini import generate_answer_stream as gemini_generate_answer_stream
from llm.openrouter import OpenRouterError, generate_answer as openrouter_generate_answer
from llm.openrouter import generate_answer_stream as openrouter_generate_answer_stream

SUPPORTED_PROVIDERS = frozenset({"gemini", "openrouter"})


@dataclass(frozen=True)
class LlmConfig:
    provider: str
    model: str
    api_key: str


def is_openrouter_key(api_key: str) -> bool:
    return api_key.strip().startswith("sk-or-")


def resolve_effective_provider(provider: str | None, api_key: str | None) -> str:
    normalized = (provider or "gemini").strip().lower() or "gemini"
    key = (api_key or "").strip()

    if normalized == "openrouter" or is_openrouter_key(key):
        return "openrouter"

    return normalized


def resolve_llm_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LlmConfig:
    resolved_model = (model or "").strip()
    if not resolved_model:
        raise ValueError("model is required (forward from backend ChatModule)")

    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise ValueError("apiKey is required (decrypted key from backend ChatModule)")

    effective_provider = resolve_effective_provider(provider, resolved_key)
    if effective_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {effective_provider}")

    return LlmConfig(
        provider=effective_provider,
        model=resolved_model,
        api_key=resolved_key,
    )


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    config = resolve_llm_config(provider=provider, model=model, api_key=api_key)

    try:
        if config.provider == "openrouter":
            return openrouter_generate_answer(
                system_prompt,
                user_prompt,
                api_key=config.api_key,
                model=config.model,
                provider=config.provider,
                history=history,
            )

        return gemini_generate_answer(
            system_prompt,
            user_prompt,
            api_key=config.api_key,
            model=config.model,
            provider=config.provider,
            history=history,
        )
    except OpenRouterError as exc:
        raise GeminiError(str(exc)) from exc


def generate_answer_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
):
    config = resolve_llm_config(provider=provider, model=model, api_key=api_key)

    try:
        if config.provider == "openrouter":
            yield from openrouter_generate_answer_stream(
                system_prompt,
                user_prompt,
                api_key=config.api_key,
                model=config.model,
                provider=config.provider,
                history=history,
            )
            return

        yield from gemini_generate_answer_stream(
            system_prompt,
            user_prompt,
            api_key=config.api_key,
            model=config.model,
            provider=config.provider,
            history=history,
        )
    except OpenRouterError as exc:
        raise GeminiError(str(exc)) from exc
