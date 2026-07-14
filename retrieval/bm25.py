from __future__ import annotations

import math
import re
from typing import Any

from db.mongo import get_database

TOKEN_RE = re.compile(r"[a-z0-9]+")


async def search_bm25_chunks(
    query_text: str,
    *,
    metadata_filter: dict[str, Any] | None = None,
    top_k: int = 5,
    db: Any | None = None,
) -> list[dict[str, Any]]:
    if not query_text or not query_text.strip():
        return []

    if db is None:
        try:
            db = get_database()
        except RuntimeError:
            return []

    query_terms = _tokenize(query_text)
    if not query_terms:
        return []

    filter_doc = _build_filter(metadata_filter)
    cursor = db.chunks.find(filter_doc, {"_id": 1, "text": 1, "metadata": 1, "articleId": 1})
    documents = await cursor.to_list(length=max(top_k * 20, 100))

    if not documents:
        return []

    article_ids = [doc.get("articleId") for doc in documents if doc.get("articleId")]
    article_fields_by_id = await _load_article_fields_batch(db, article_ids)

    enriched_documents: list[dict[str, Any]] = []
    for doc in documents:
        text = (doc.get("text") or "").strip()
        if not text:
            continue

        metadata = doc.get("metadata") or {}
        article_key = str(doc.get("articleId", ""))
        article_fields = article_fields_by_id.get(article_key, {})
        if article_fields:
            metadata = {**metadata, **article_fields}

        enriched_documents.append(
            {
                "chunk_id": str(doc.get("_id", "")),
                "text": text,
                "source": str(metadata.get("source") or ""),
                "title": str(metadata.get("title") or ""),
                "url": str(metadata.get("url") or ""),
                "published_at": str(metadata.get("publishedAt") or ""),
                "metadata": metadata,
            }
        )

    if not enriched_documents:
        return []

    scores = _score_bm25_corpus([entry["text"] for entry in enriched_documents], query_terms)
    scored_documents: list[tuple[float, dict[str, Any]]] = []
    for score, entry in zip(scores, enriched_documents):
        if score <= 0:
            continue
        scored_documents.append(
            (
                score,
                {
                    **entry,
                    "bm25_score": score,
                },
            )
        )

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in scored_documents[: max(1, top_k)]]


def _format_article_fields(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": article.get("title") or "",
        "url": article.get("url") or "",
        "source": article.get("source") or "",
        "publishedAt": article.get("publishedAt") or "",
        "category": article.get("category") or "",
        "author": article.get("author") or "",
    }


async def _load_article_fields_batch(db: Any, article_ids: list[Any]) -> dict[str, dict[str, Any]]:
    unique_ids = list({article_id for article_id in article_ids if article_id})
    if not unique_ids:
        return {}

    try:
        cursor = db.news_articles.find(
            {"_id": {"$in": unique_ids}},
            {"title": 1, "url": 1, "source": 1, "publishedAt": 1, "category": 1, "author": 1},
        )
        articles = await cursor.to_list(length=len(unique_ids))
    except Exception:
        return {}

    return {str(article["_id"]): _format_article_fields(article) for article in articles}


def _build_filter(metadata_filter: dict[str, Any] | None) -> dict[str, Any]:
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


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token]


def _score_bm25_corpus(texts: list[str], query_terms: list[str]) -> list[float]:
    if not texts:
        return []

    try:
        from rank_bm25 import BM25Okapi
    except Exception:
        return [_simple_bm25(text, query_terms) for text in texts]

    tokenized_documents = [_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(query_terms)
    return [float(score) for score in scores]


def _simple_bm25(text: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    tokenized_text = _tokenize(text)
    if not tokenized_text:
        return 0.0

    term_freq = {term: tokenized_text.count(term) for term in set(query_terms)}
    doc_len = len(tokenized_text)
    avg_doc_len = max(doc_len, 1)
    score = 0.0
    for term, freq in term_freq.items():
        if freq <= 0:
            continue
        idf = 1.0 + math.log((1 + 1) / (1 + 1))
        tf = (freq * (1.2 + 1)) / (freq + 1.2 * (1 - 0.75 + 0.75 * (doc_len / avg_doc_len)))
        score += idf * tf
    return score
