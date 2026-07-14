from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from db.mongo import get_database


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    title: str
    url: str
    published_at: str
    similarity_score: float


def get_vector_index_name() -> str:
    return os.getenv("VECTOR_SEARCH_INDEX", "chunks_vector_index").strip() or "chunks_vector_index"


def get_default_top_k() -> int:
    raw = os.getenv("RAG_TOP_K", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        return 5
    return max(1, min(value, 20))


def _format_published_at(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    if value is None:
        return ""
    return str(value)


async def search_similar_chunks(
    query_embedding: list[float],
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not query_embedding:
        return []

    try:
        db = get_database()
    except RuntimeError:
        return []

    limit = top_k or get_default_top_k()
    index_name = get_vector_index_name()
    vector_filter = _build_vector_filter(metadata_filter)

    vector_stage: dict[str, Any] = {
        "index": index_name,
        "path": "embedding",
        "queryVector": query_embedding,
        "numCandidates": max(limit * 10, 50),
        "limit": limit,
    }
    if vector_filter:
        vector_stage["filter"] = vector_filter

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": vector_stage
        },
        {
            "$lookup": {
                "from": "news_articles",
                "localField": "articleId",
                "foreignField": "_id",
                "as": "article",
            }
        },
        {"$unwind": {"path": "$article", "preserveNullAndEmptyArrays": True}},
        {
            "$project": {
                "_id": 1,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
                "source": {
                    "$ifNull": ["$article.source", {"$ifNull": ["$metadata.source", ""]}]
                },
                "title": {"$ifNull": ["$article.title", ""]},
                "url": {"$ifNull": ["$article.url", ""]},
                "publishedAt": {
                    "$ifNull": [
                        "$article.publishedAt",
                        {"$ifNull": ["$metadata.publishedAt", ""]},
                    ]
                },
            }
        },
    ]

    try:
        cursor = db.chunks.aggregate(pipeline)
        documents = await cursor.to_list(length=limit)
    except Exception:
        return []

    results: list[RetrievedChunk] = []
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        results.append(
            RetrievedChunk(
                chunk_id=str(doc.get("_id", "")),
                text=text,
                source=str(doc.get("source") or ""),
                title=str(doc.get("title") or ""),
                url=str(doc.get("url") or ""),
                published_at=_format_published_at(doc.get("publishedAt")),
                similarity_score=float(doc.get("score") or 0.0),
            )
        )

    return results


def _build_vector_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_filter:
        return {}

    filter_doc: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if key == "publishedAt":
            filter_doc["metadata.publishedAt"] = value
        elif key == "date":
            continue
        elif key == "author":
            filter_doc["metadata.author"] = {
                "$regex": f"^{re.escape(str(value))}$",
                "$options": "i",
            }
        else:
            filter_doc[f"metadata.{key}"] = {"$eq": value}
    return filter_doc
