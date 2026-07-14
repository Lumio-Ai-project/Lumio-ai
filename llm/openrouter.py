from __future__ import annotations

import logging
import os


class OpenRouterError(Exception):
    """Raised when the OpenRouter API call fails."""


OPENROUTER_BASE_URL = (
    os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    or "https://openrouter.ai/api/v1"
)


# Maps deprecated or shorthand OpenRouter model IDs to current valid equivalents.
# Entries must use the fully-qualified "provider/model-id" form.
_MODEL_ALIASES: dict[str, str] = {
    # gemini-2.0-flash was never a valid OpenRouter ID; the versioned form
    # (google/gemini-2.0-flash-001) was deprecated on 2025-06-01 and removed.
    # Redirect to a free model so requests succeed without requiring credits.
    "google/gemini-2.0-flash": "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-001": "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-lite": "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-lite-001": "meta-llama/llama-3.3-70b-instruct:free",
    # gemini-1.5 aliases for forward-compat
    "google/gemini-1.5-flash": "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-1.5-pro": "meta-llama/llama-3.3-70b-instruct:free",
}


def normalize_openrouter_model(model: str) -> str:
    trimmed = model.strip()
    if not trimmed:
        raise ValueError("model is required (forward from backend ChatModule)")

    # Expand shorthand names that have no provider prefix
    if "/" not in trimmed:
        if trimmed.startswith("gemini-"):
            trimmed = f"google/{trimmed}"
        elif trimmed.startswith("gemma-"):
            trimmed = f"google/{trimmed}"

    # Redirect deprecated / invalid IDs to current equivalents
    resolved = _MODEL_ALIASES.get(trimmed, trimmed)
    if resolved != trimmed:
        logging.getLogger(__name__).warning(
            "OpenRouter model '%s' is deprecated or invalid; redirecting to '%s'.",
            trimmed,
            resolved,
        )

    return resolved


def reasoning_enabled(model: str) -> bool:
    override = os.getenv("OPENROUTER_REASONING", "").strip().lower()
    if override in {"1", "true", "yes", "on"}:
        return True
    if override in {"0", "false", "no", "off"}:
        return False

    # RAG answers need final text in `content`; reasoning-only models often leave it empty.
    return False


def _extract_message_content(message: object) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return "\n".join(parts)

    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    if isinstance(reasoning, list):
        parts = [part.strip() for part in reasoning if isinstance(part, str) and part.strip()]
        if parts:
            return "\n".join(parts)

    if isinstance(reasoning, dict):
        text = reasoning.get("content") or reasoning.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning", "reasoning_content"):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    raise OpenRouterError("OpenRouter returned an empty response")


def get_max_tokens() -> int:
    raw = os.getenv("OPENROUTER_MAX_TOKENS", "1024").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1024
    return max(256, min(value, 8192))


def _is_insufficient_credits_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "402" in message and ("credit" in message or "max_tokens" in message)


def _format_openrouter_error(exc: Exception) -> str:
    message = str(exc)
    if _is_insufficient_credits_error(exc):
        return (
            "OpenRouter credits are insufficient for this request. "
            "Add credits at https://openrouter.ai/settings/credits "
            "or set OPENROUTER_MAX_TOKENS to a lower value (e.g. 512)."
        )
    return message


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    del provider  # routing handled by llm.client

    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise ValueError("apiKey is required (decrypted key from backend ChatModule)")

    resolved_model = normalize_openrouter_model(model or "")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenRouterError(
            "openai package is required for OpenRouter. Run: pip install openai"
        ) from exc

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=resolved_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:5173"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Lumio"),
        },
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    request_kwargs: dict[str, object] = {
        "model": resolved_model,
        "messages": messages,
    }

    if reasoning_enabled(resolved_model):
        request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    max_tokens = get_max_tokens()
    response = None

    while max_tokens >= 256:
        request_kwargs["max_tokens"] = max_tokens
        try:
            response = client.chat.completions.create(**request_kwargs)
            break
        except Exception as exc:
            if _is_insufficient_credits_error(exc) and max_tokens > 256:
                max_tokens = max(256, max_tokens // 2)
                continue
            raise OpenRouterError(_format_openrouter_error(exc)) from exc

    if response is None:
        raise OpenRouterError(
            "OpenRouter credits are insufficient for this request. "
            "Add credits at https://openrouter.ai/settings/credits "
            "or set OPENROUTER_MAX_TOKENS to a lower value (e.g. 512)."
        )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise OpenRouterError("OpenRouter returned an empty response")

    return _extract_message_content(choices[0].message)


def generate_answer_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
):
    """Yield text chunks from OpenRouter streaming generation."""
    del provider

    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise ValueError("apiKey is required (decrypted key from backend ChatModule)")

    resolved_model = normalize_openrouter_model(model or "")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise OpenRouterError(
            "openai package is required for OpenRouter. Run: pip install openai"
        ) from exc

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=resolved_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:5173"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "Lumio"),
        },
    )

    request_kwargs: dict[str, object] = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *([] if not history else history),
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
        "max_tokens": get_max_tokens(),
    }

    if reasoning_enabled(resolved_model):
        request_kwargs["extra_body"] = {"reasoning": {"enabled": True}}

    try:
        stream = client.chat.completions.create(**request_kwargs)
    except Exception as exc:
        raise OpenRouterError(_format_openrouter_error(exc)) from exc

    yielded_any = False
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            yielded_any = True
            yield content

    if not yielded_any:
        raise OpenRouterError("OpenRouter returned an empty response")
