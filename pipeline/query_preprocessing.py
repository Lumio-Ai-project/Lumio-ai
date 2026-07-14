from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from llm.gemini import GeminiError, generate_answer


@dataclass(frozen=True)
class QueryPreprocessingResult:
    intent: str
    rewritten_query: str
    filters: dict[str, Any]
    is_out_of_scope: bool


SOURCE_ALIASES = {
    "reuters": "Reuters",
    "bbc": "BBC",
    "techcrunch": "TechCrunch",
    "theverge": "The Verge",
    "tnw": "TNW",
    "venturebeat": "VentureBeat",
    "sciencedaily": "ScienceDaily",
    "physorg": "Phys.org",
    "openai": "OpenAI",
    "huggingface": "Hugging Face",
    "apnews": "AP News",
}

CATEGORY_ALIASES = {
    "ai": "AI",
    "artificial intelligence": "AI",
    "technology": "Technology",
    "tech": "Technology",
    "science": "Science",
    "general": "General",
}


def preprocess_query(question: str, llm: Any | None = None) -> QueryPreprocessingResult:
    trimmed = (question or "").strip()
    if not trimmed:
        return QueryPreprocessingResult(
            intent="General Question",
            rewritten_query="",
            filters={},
            is_out_of_scope=True,
        )

    intent = classify_query(trimmed)
    filters = extract_filters(trimmed)
    is_out_of_scope = detect_out_of_scope(trimmed)
    rewritten_query = rewrite_query(trimmed, llm=llm)

    if is_out_of_scope:
        rewritten_query = trimmed

    return QueryPreprocessingResult(
        intent=intent,
        rewritten_query=rewritten_query or trimmed,
        filters=filters,
        is_out_of_scope=is_out_of_scope,
    )


def classify_query(question: str) -> str:
    lowered = question.lower()

    if any(word in lowered for word in ["latest", "new", "recent", "today", "yesterday", "this week", "this month"]):
        return "Latest News"
    if any(word in lowered for word in ["summarize", "summary", "overview"]):
        return "Summary"
    if any(word in lowered for word in ["compare", "vs", "versus", "difference between"]):
        return "Comparison"
    if any(word in lowered for word in ["from yesterday", "from today", "last week", "last month", "on ", "published"]):
        return "Date Query"
    if any(word in lowered for word in ["category", "in ai", "in science", "in technology", "ai news", "science news"]):
        return "Category Query"
    if any(word in lowered for word in ["from reuters", "from bbc", "from techcrunch", "source", "published by"]):
        return "Source Query"
    if any(word in lowered for word in ["what", "why", "how", "who", "when", "can you"]):
        return "General Question"
    return "General Question"


def extract_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    source = _find_source(question)
    if source:
        filters["source"] = source

    category = _find_category(question)
    if category:
        filters["category"] = category

    date = _find_date(question)
    if date:
        filters["date"] = date

    language = _find_language(question)
    if language:
        filters["language"] = language

    author = _find_author(question)
    if author:
        filters["author"] = author

    return filters


def build_metadata_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None

    metadata_filter: dict[str, Any] = {}
    for key, value in filters.items():
        if key == "date":
            metadata_filter["publishedAt"] = _resolve_date_filter(value)
        elif key == "source":
            metadata_filter["source"] = value
        elif key == "category":
            metadata_filter["category"] = value
        elif key == "language":
            metadata_filter["language"] = value
        elif key == "author":
            metadata_filter["author"] = value

    return metadata_filter or None


def rewrite_query(question: str, llm: Any | None = None) -> str:
    lowered = question.lower().strip()
    if any(token in lowered for token in ["what's new", "latest", "recent", "news", "today", "yesterday"]):
        if "artificial intelligence" in lowered or "ai" in lowered:
            return "Latest Artificial Intelligence news published recently"
        if "technology" in lowered:
            return "Latest technology news published recently"
        if "science" in lowered:
            return "Latest science news published recently"

    if llm is None:
        return question

    try:
        prompt = (
            "Rewrite the following news question into a clearer search query for a news retrieval system. "
            "Keep it short and specific.\n\nQuestion: "
            f"{question}"
        )
        response = generate_answer(
            "You rewrite search queries for a news retrieval system.",
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


def detect_out_of_scope(question: str) -> bool:
    lowered = question.lower()
    if any(token in lowered for token in ["weather", "recipe", "translate", "code", "python", "javascript", "math", "capital of"]):
        return True
    if any(token in lowered for token in ["news", "article", "article", "source", "published", "latest"]):
        return False
    return False


def _find_source(question: str) -> str | None:
    lowered = question.lower()
    for alias, canonical in SOURCE_ALIASES.items():
        if alias in lowered:
            return canonical
    return None


def _find_category(question: str) -> str | None:
    lowered = question.lower()
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in lowered:
            return canonical
    return None


def _find_date(question: str) -> str | None:
    lowered = question.lower()
    if "today" in lowered:
        return "today"
    if "yesterday" in lowered:
        return "yesterday"
    if "last week" in lowered:
        return "last week"
    if "last month" in lowered:
        return "last month"
    if "this week" in lowered:
        return "this week"
    if "this month" in lowered:
        return "this month"
    return None


def _find_language(question: str) -> str | None:
    lowered = question.lower()
    if "english" in lowered:
        return "en"
    if "arabic" in lowered:
        return "ar"
    return None


def _find_author(question: str) -> str | None:
    match = re.search(r"(?:by|written by)\s+([A-Za-z][A-Za-z .'-]+)", question, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _resolve_date_filter(value: Any) -> dict[str, Any] | None:
    if not value:
        return None

    if not isinstance(value, str):
        return None

    now = datetime.now(timezone.utc)
    if value == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return {"$gte": start.isoformat(), "$lte": now.isoformat()}
    if value == "yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = (now - timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)
        return {"$gte": start.isoformat(), "$lte": end.isoformat()}
    if value == "last week":
        start = now - timedelta(days=7)
        return {"$gte": start.isoformat(), "$lte": now.isoformat()}
    if value == "last month":
        start = now - timedelta(days=30)
        return {"$gte": start.isoformat(), "$lte": now.isoformat()}
    return {"$regex": re.escape(value), "$options": "i"}
