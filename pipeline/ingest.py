from __future__ import annotations

from typing import Any

from bson import ObjectId

from db.mongo import get_database, init_mongo
from preprocessing.chunker import chunk_article
from preprocessing.cleaner import clean_text
from scraper.base import Article
from scraper.registry import get_scraper, list_sources


async def run_ingest(
    sources: list[str] | None = None,
    limit: int = 50,
    dry_run: bool = False,
) -> dict[str, Any]:
    selected_sources = sources or list_sources()
    articles_ingested = 0
    chunks_created = 0
    errors: list[str] = []

    db = None
    if not dry_run:
        db = await init_mongo()

    for source_name in selected_sources:
        try:
            scraper = get_scraper(source_name)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        scrape_result = scraper.scrape(limit=limit)
        errors.extend(scrape_result.errors)

        for article in scrape_result.articles:
            article = _normalize_article(article)
            if dry_run:
                chunks = _build_chunks(article, article_id=ObjectId())
                articles_ingested += 1
                chunks_created += len(chunks)
                continue

            stored, created_chunks = await _store_article(db, article)
            if stored:
                articles_ingested += 1
                chunks_created += created_chunks

    return {
        "articlesIngested": articles_ingested,
        "chunksCreated": chunks_created,
        "sources": selected_sources,
        "errors": errors,
        "dryRun": dry_run,
    }


def _normalize_article(article: Article) -> Article:
    return Article(
        title=clean_text(article.title),
        content=clean_text(article.content),
        summary=clean_text(article.summary) if article.summary else None,
        category=article.category,
        author=article.author,
        published_at=article.published_at,
        source=article.source,
        url=article.url.strip(),
        language=article.language or "en",
    )


def _build_chunks(article: Article, article_id: ObjectId) -> list[dict]:
    metadata = {
        "category": article.category,
        "source": article.source,
        "publishedAt": article.published_at,
        "language": article.language,
    }
    chunks = chunk_article(article.content, metadata=metadata)
    return [
        {
            "articleId": article_id,
            "chunkIndex": chunk.chunk_index,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


async def _store_article(db, article: Article) -> tuple[bool, int]:
    existing = await db.news_articles.find_one({"url": article.url}, {"_id": 1})
    if existing:
        return False, 0

    insert_result = await db.news_articles.insert_one(article.to_dict())
    article_id = insert_result.inserted_id
    chunk_docs = _build_chunks(article, article_id)

    if chunk_docs:
        await db.chunks.insert_many(chunk_docs)

    return True, len(chunk_docs)
