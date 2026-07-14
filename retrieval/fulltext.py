from __future__ import annotations

import re
from typing import Any

from db.mongo import get_database
from retrieval.vector_search import RetrievedChunk, _format_published_at


async def search_fulltext_chunks(
    query_text: str,
    *,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not query_text or not query_text.strip():
        return []

    try:
        db = get_database()
    except RuntimeError:
        return []

    limit = top_k or 5
    filter_doc = _build_text_filter(metadata_filter)

    pipeline: list[dict[str, Any]] = [
        {
            "$match": {
                "$text": {"$search": query_text},
                **filter_doc,
            }
        },
        {
            "$addFields": {"textScore": {"$meta": "textScore"}}
        },
        {"$sort": {"textScore": -1}},
        {"$limit": limit},
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
                "textScore": 1,
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

    if not documents:
        return []

    max_score = max(doc.get("textScore", 0) for doc in documents) or 1.0

    results: list[RetrievedChunk] = []
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        raw_score = float(doc.get("textScore") or 0.0)
        normalized = min(1.0, raw_score / max_score) if max_score > 0 else 0.0

        results.append(
            RetrievedChunk(
                chunk_id=str(doc.get("_id", "")),
                text=text,
                source=str(doc.get("source") or ""),
                title=str(doc.get("title") or ""),
                url=str(doc.get("url") or ""),
                published_at=_format_published_at(doc.get("publishedAt")),
                similarity_score=round(normalized, 6),
            )
        )

    return results


def _build_text_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata_filter:
        return {}

    filter_doc: dict[str, Any] = {}
    for key, value in metadata_filter.items():
        if key == "date" and isinstance(value, dict):
            filter_doc["metadata.publishedAt"] = value
            continue
        if key == "publishedAt":
            filter_doc["metadata.publishedAt"] = value
            continue
        filter_doc[f"metadata.{key}"] = value
    return filter_doc
