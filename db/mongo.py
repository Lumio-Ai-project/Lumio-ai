import os
from typing import Optional
from urllib.parse import unquote, urlparse

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


def get_mongo_uri() -> str:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    return uri


def get_database_name() -> str:
    """Resolve DB name: MONGODB_DB env → path in MONGODB_URI → default."""
    explicit = os.getenv("MONGODB_DB", "").strip()
    if explicit:
        return explicit

    uri = os.getenv("MONGODB_URI", "").strip()
    if uri:
        from_path = _database_name_from_uri(uri)
        if from_path:
            return from_path

    return "news_rag"


def _database_name_from_uri(uri: str) -> str | None:
    """Extract database name from URI path (same as Mongoose default)."""
    normalized = uri.replace("mongodb+srv://", "https://").replace("mongodb://", "http://")
    path = urlparse(normalized).path.strip("/")
    if not path:
        return None
    # Ignore query-only paths; take first segment only.
    database = unquote(path.split("/")[0]).strip()
    return database or None


async def init_mongo() -> AsyncIOMotorDatabase:
    global _client, _database

    if _database is not None:
        return _database

    _client = AsyncIOMotorClient(get_mongo_uri())
    _database = _client[get_database_name()]

    await _database.command("ping")
    await _ensure_indexes(_database)
    return _database


async def close_mongo() -> None:
    global _client, _database

    if _client is not None:
        _client.close()

    _client = None
    _database = None


def get_database() -> AsyncIOMotorDatabase:
    if _database is None:
        raise RuntimeError("MongoDB is not initialized. Call init_mongo() first.")
    return _database


async def _ensure_indexes(db: AsyncIOMotorDatabase) -> None:
    await db.news_articles.create_index("url", unique=True, name="url_unique")
    await db.news_articles.create_index([("publishedAt", -1)], name="publishedAt_desc")
    await db.news_articles.create_index(
        [("source", 1), ("category", 1)],
        name="source_category",
    )
    await db.chunks.create_index(
        [("articleId", 1), ("chunkIndex", 1)],
        name="article_chunk",
    )
    await db.chunks.create_index(
        [("text", "text")],
        name="chunks_text",
    )
    # Atlas Vector Search index on chunks.embedding must be created in the Atlas UI
    # (384 dimensions, cosine similarity). See ai-service/README.md Phase 5 section.
