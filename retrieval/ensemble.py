from __future__ import annotations

import asyncio
import os
from typing import Any

from retrieval.bm25 import search_bm25_chunks
from retrieval.fulltext import search_fulltext_chunks
from retrieval.vector_search import RetrievedChunk, search_similar_chunks


def _rrf_k() -> int:
    raw = os.getenv("RRF_K", "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


async def search_ensemble_chunks(
    query_embedding: list[float],
    query_text: str,
    *,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    if not query_embedding or not query_text.strip():
        return []

    limit = max(1, top_k or 5)
    per_retriever_limit = max(limit * 3, 15)
    k = _rrf_k()

    vector_task = search_similar_chunks(
        query_embedding,
        top_k=per_retriever_limit,
        metadata_filter=metadata_filter,
    )
    bm25_task = search_bm25_chunks(
        query_text,
        top_k=per_retriever_limit,
        metadata_filter=metadata_filter,
    )
    fulltext_task = search_fulltext_chunks(
        query_text,
        top_k=per_retriever_limit,
        metadata_filter=metadata_filter,
    )

    vector_results, bm25_results, fulltext_results = await asyncio.gather(
        vector_task, bm25_task, fulltext_task,
    )

    if not vector_results and not bm25_results and not fulltext_results:
        return []

    all_lists: list[list[RetrievedChunk]] = []

    all_lists.append(vector_results)

    bm25_chunks: list[RetrievedChunk] = []
    for item in bm25_results:
        bm25_chunks.append(
            RetrievedChunk(
                chunk_id=str(item.get("chunk_id", "")),
                text=str(item.get("text", "")),
                source=str(item.get("source", "")),
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                published_at=str(item.get("published_at", "")),
                similarity_score=float(item.get("bm25_score", 0.0)),
            )
        )
    all_lists.append(bm25_chunks)

    all_lists.append(fulltext_results)

    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for retriever_list in all_lists:
        for rank, chunk in enumerate(retriever_list):
            key = chunk.chunk_id or chunk.url or chunk.text
            if not key:
                continue

            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

            if key not in chunk_map:
                chunk_map[key] = chunk

    scored_chunks = sorted(
        [(score, chunk_map[key]) for key, score in rrf_scores.items()],
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            title=chunk.title,
            url=chunk.url,
            published_at=chunk.published_at,
            similarity_score=round(score, 6),
        )
        for score, chunk in scored_chunks[:limit]
    ]
