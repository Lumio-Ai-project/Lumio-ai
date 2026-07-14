from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = frozenset({"gemini"})


class GeminiError(Exception):
    """Raised when the Gemini API call fails."""


@dataclass(frozen=True)
class GeminiConfig:
    provider: str
    model: str
    api_key: str


def normalize_gemini_model(model: str) -> str:
    """Strip OpenRouter-style prefixes/suffixes so the Gemini SDK receives a clean model name.

    Examples:
        "google/gemma-3-27b-it:free"  -> "gemma-3-27b-it"
        "google/gemini-1.5-flash"     -> "gemini-1.5-flash"
        "gemini-2.0-flash"            -> "gemini-2.0-flash"
    """
    trimmed = model.strip()
    # Strip provider prefix added by OpenRouter (e.g. "google/")
    if "/" in trimmed:
        trimmed = trimmed.split("/", 1)[1]
    # Strip OpenRouter tier suffix (e.g. ":free", ":paid", ":nitro")
    if ":" in trimmed:
        trimmed = trimmed.split(":")[0]
    return trimmed.strip()


def resolve_gemini_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> GeminiConfig:
    resolved_provider = (provider or "gemini").strip().lower() or "gemini"
    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {resolved_provider}")

    resolved_model = normalize_gemini_model((model or "").strip())
    if not resolved_model:
        raise ValueError("model is required (forward from backend ChatModule)")

    resolved_key = (api_key or "").strip()
    if not resolved_key:
        raise ValueError("apiKey is required (decrypted key from backend ChatModule)")

    return GeminiConfig(
        provider=resolved_provider,
        model=resolved_model,
        api_key=resolved_key,
    )


def _build_gemini_contents(
    history: list[dict[str, str]] | None,
    user_prompt: str,
) -> list[dict[str, str]] | str:
    if not history:
        return user_prompt

    contents: list[dict[str, str]] = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [msg["content"]]})
    contents.append({"role": "user", "parts": [user_prompt]})
    return contents


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    config = resolve_gemini_config(provider=provider, model=model, api_key=api_key)

    try:
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        gemini_model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=system_prompt,
        )
        contents = _build_gemini_contents(history, user_prompt)
        response = gemini_model.generate_content(contents)
    except RuntimeError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise GeminiError(str(exc)) from exc

    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()

    raise GeminiError("Gemini returned an empty response")


def generate_answer_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
):
    """Yield text chunks from Gemini streaming generation."""
    config = resolve_gemini_config(provider=provider, model=model, api_key=api_key)

    try:
        import google.generativeai as genai

        genai.configure(api_key=config.api_key)
        gemini_model = genai.GenerativeModel(
            model_name=config.model,
            system_instruction=system_prompt,
        )
        contents = _build_gemini_contents(history, user_prompt)
        stream = gemini_model.generate_content(contents, stream=True)
    except RuntimeError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise GeminiError(str(exc)) from exc

    yielded_any = False
    try:
        for chunk in stream:
            text = getattr(chunk, "text", None)
            if text:
                yielded_any = True
                yield text
    except Exception as exc:
        raise GeminiError(str(exc)) from exc

    if not yielded_any:
        raise GeminiError("Gemini returned an empty response")
