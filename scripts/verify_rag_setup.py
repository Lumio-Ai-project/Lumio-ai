"""Verify MongoDB RAG preflight: chunks, embeddings, Atlas vector index."""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from db.mongo import close_mongo, init_mongo


async def main() -> None:
    load_dotenv()
    db = await init_mongo()

    total = await db.chunks.count_documents({})
    embedded = await db.chunks.count_documents({"embedding": {"$exists": True}})
    articles = await db.news_articles.count_documents({})

    print(f"articles={articles} chunks={total} embedded={embedded}")

    sample = await db.chunks.find_one(
        {"embedding": {"$exists": True}},
        {"_id": 1, "embedding": 1},
    )
    if sample and sample.get("embedding"):
        print(f"sample_embedding_dims={len(sample['embedding'])}")

    indexes = await db.chunks.list_search_indexes().to_list(length=10)
    for index in indexes:
        print(
            "index:",
            index.get("name"),
            "status=",
            index.get("status"),
            "queryable=",
            index.get("queryable"),
        )

    await close_mongo()


if __name__ == "__main__":
    asyncio.run(main())
