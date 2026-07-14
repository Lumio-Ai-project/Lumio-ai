from __future__ import annotations

import os
import re
from typing import Any

from llm.gemini import GeminiError, generate_answer

HISTORY_MESSAGE = dict[str, str]
HistoryMessage = HISTORY_MESSAGE

_ALEF_VARIANTS = ("أ", "إ", "آ", "ٱ")

_NO_RETRIEVAL_PATTERNS = (
    r"\bmy name is\b",
    r"\bcall me\b",
    r"\bwhat(?:'s| is) my name\b",
    r"\bwho am i\b",
    r"\bwhat did i (?:say|ask|tell|mention)\b",
    r"\bdo you remember\b",
    r"\bhow are you\b",
    r"^(?:hi|hello|hey|thanks|thank you|ok|okay)\b",
    r"اسمي\s+ا(?:يه|ي)",
    r"ما\s+(?:هو|هي)\s+اسمي",
    r"ما\s+اسمي",
    r"اسمي\s+ه(?:و|ي)",
    r"انا\s+اسمي",
    r"مين\s+انا",
    r"(?:مش|لا)\s+ع(?:ا|أ)يز(?:\s+ال)?\s*(?:source|resourc|sources|مصدر|مصادر)",
    r"(?:ماذا|ما)\s+(?:هي\s+)?(?:ال)?اس(?:ئلة|ئله|اله)",
    r"(?:ال)?اس(?:ئلة|ئله|اله)\s+(?:التي|اللي)\s+(?:سأ?لت(?:ه)?(?:ال)?ك?)?",
    r"ماذا\s+سأ?لت(?:ك)?",
    r"اذكر\s+ما\s+(?:قلت|سأ?لت)",
)

_CONVERSATIONAL_FOLLOWUP_PATTERNS = _NO_RETRIEVAL_PATTERNS + (
    r"\bwhat was my\b",
    r"\b(remember|recall).+(?:name|said|asked|told)\b",
    r"فاكر\s+اسمي",
    r"ت(?:ت)?ذكر\s+",
)


def normalize_question_text(text: str) -> str:
    normalized = text.strip()
    for variant in _ALEF_VARIANTS:
        normalized = normalized.replace(variant, "ا")
    return normalized

_NEWS_INTENT_PATTERNS = (
    r"\bnews\b",
    r"\blatest\b",
    r"\brecent\b",
    r"\bheadline\b",
    r"\barticle\b",
    r"\bbreaking\b",
    r"\bwhat happened\b",
    r"\bwhat(?:'s| is) happening\b",
    r"\btell me about\b",
    r"\bsummarize\b",
    r"\bsummary\b",
    r"\bupdate on\b",
    r"\bupdates on\b",
    r"\breport on\b",
    r"\b(reuters|bbc|techcrunch|the verge|ap news)\b",
)

_NEWS_FOLLOWUP_PATTERNS = (
    r"\b(it|that|this|they|them|those)\b",
    r"\bmore about\b",
    r"\bmore on\b",
    r"\bwhat about\b",
    r"\bhow about\b",
)


def is_conversational_followup(question: str, history: list[HISTORY_MESSAGE]) -> bool:
    """True when the user refers to prior chat state rather than external news."""
    if not history:
        return False

    normalized = normalize_question_text(question)
    if not normalized:
        return False

    for pattern in _CONVERSATIONAL_FOLLOWUP_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return True

    return False


def _matches_any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _is_news_followup(question: str, history: list[HISTORY_MESSAGE]) -> bool:
    if not history or not _matches_any(_NEWS_FOLLOWUP_PATTERNS, question):
        return False

    recent_text = " ".join(msg.get("content", "") for msg in history[-6:])
    return _matches_any(_NEWS_INTENT_PATTERNS, recent_text)


def should_retrieve_news(question: str, history: list[HISTORY_MESSAGE]) -> bool:
    """Return True only when the question needs news article retrieval."""
    normalized = normalize_question_text(question)
    if not normalized:
        return False

    if _matches_any(_NO_RETRIEVAL_PATTERNS, normalized):
        return False

    if is_conversational_followup(normalized, history):
        return False

    if _matches_any(_NEWS_INTENT_PATTERNS, normalized):
        return True

    if _is_news_followup(normalized, history):
        return True

    # Short messages without news intent are treated as chat (e.g. introductions).
    if len(normalized.split()) <= 10:
        return False

    return False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def get_history_max_tokens() -> int:
    raw = os.getenv("HISTORY_MAX_TOKENS", "2000").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 2000


def history_total_tokens(history: list[HISTORY_MESSAGE]) -> int:
    return sum(estimate_tokens(msg.get("content", "")) for msg in history)


def summarize_history(
    history: list[HISTORY_MESSAGE],
    *,
    llm: Any | None = None,
    max_tokens: int | None = None,
) -> list[HISTORY_MESSAGE]:
    if not history:
        return []

    budget = max_tokens if max_tokens is not None else get_history_max_tokens()
    if budget <= 0:
        return []

    if history_total_tokens(history) <= budget:
        return list(history)

    if llm is None:
        return _keep_newest_within_budget(history, budget)

    mid = len(history) // 2
    older = history[:mid]
    newer = history[mid:]

    summary_text = _summarize_via_llm(older, llm)
    if summary_text:
        return [
            {"role": "assistant", "content": f"[Previous conversation summary]\n{summary_text}"},
            *newer,
        ]

    return _keep_newest_within_budget(history, budget)


def rewrite_query_with_history(
    question: str,
    history: list[HISTORY_MESSAGE],
    *,
    llm: Any | None = None,
) -> str:
    if not history or llm is None:
        return question

    recent = history[-6:]
    turns_text = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in recent
    )

    prompt = (
        "Rewrite the following user question as a standalone search query for a news retrieval system. "
        "Use the conversation history to resolve any pronouns (it, they, that, them) or "
        "references to earlier topics. Keep the rewritten query short and specific.\n\n"
        f"Conversation history:\n{turns_text}\n\n"
        f"Current user question: {question}\n\n"
        "Standalone search query:"
    )

    try:
        response = generate_answer(
            "You rewrite news search queries to be standalone and self-contained.",
            prompt,
            provider=llm.provider,
            model=llm.model,
            api_key=llm.api_key,
        )
        if response and response.strip():
            return response.strip()
    except (GeminiError, ValueError, RuntimeError):
        pass

    return question


def _summarize_via_llm(history: list[HISTORY_MESSAGE], llm: Any) -> str:
    turns_text = "\n".join(
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in history
    )

    prompt = (
        "Summarize the following conversation into a concise paragraph. "
        "Focus on the key topics, questions asked, and answers given. "
        "Keep it under 150 words.\n\n"
        f"Conversation:\n{turns_text}\n\n"
        "Summary:"
    )

    try:
        response = generate_answer(
            "You summarize news conversations concisely.",
            prompt,
            provider=llm.provider,
            model=llm.model,
            api_key=llm.api_key,
        )
        if response and response.strip():
            return response.strip()
    except (GeminiError, ValueError, RuntimeError):
        pass

    return ""


def _keep_newest_within_budget(
    history: list[HISTORY_MESSAGE],
    max_tokens: int,
) -> list[HISTORY_MESSAGE]:
    result: list[HISTORY_MESSAGE] = []
    used = 0

    for msg in reversed(history):
        cost = estimate_tokens(msg.get("content", ""))
        if used + cost > max_tokens:
            break
        result.append(msg)
        used += cost

    result.reverse()
    return result
