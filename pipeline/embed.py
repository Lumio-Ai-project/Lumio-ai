from __future__ import annotations

from typing import Any

from bson import ObjectId

from db.mongo import get_database, init_mongo
from embeddings.embedding import embed_texts


async def run_embed(batch_size: int = 64) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    db = await init_mongo()
    query = {
        "$or": [
            {"embedding": {"$exists": False}},
            {"embedding": None},
            {"embedding": []},
        ]
    }

    chunks_embedded = 0
    chunks_skipped = 0
    errors: list[str] = []

    cursor = db.chunks.find(query, {"_id": 1, "text": 1})
    batch: list[dict[str, Any]] = []

    async for document in cursor:
        text = (document.get("text") or "").strip()
        if not text:
            chunks_skipped += 1
            continue

        batch.append(document)
        if len(batch) >= batch_size:
            embedded, batch_errors = await _embed_batch(db, batch)
            chunks_embedded += embedded
            errors.extend(batch_errors)
            batch = []

    if batch:
        embedded, batch_errors = await _embed_batch(db, batch)
        chunks_embedded += embedded
        errors.extend(batch_errors)

    return {
        "chunksEmbedded": chunks_embedded,
        "chunksSkipped": chunks_skipped,
        "errors": errors,
    }


async def _embed_batch(
    db,
    batch: list[dict[str, Any]],
) -> tuple[int, list[str]]:
    texts = [(doc.get("text") or "").strip() for doc in batch]
    try:
        vectors = embed_texts(texts)
    except Exception as exc:
        return 0, [f"Failed to embed batch of {len(batch)} chunks: {exc}"]

    embedded = 0
    errors: list[str] = []

    for document, vector in zip(batch, vectors, strict=True):
        chunk_id = document.get("_id")
        if not isinstance(chunk_id, ObjectId):
            errors.append(f"Skipped chunk with invalid id: {chunk_id!r}")
            continue

        await db.chunks.update_one(
            {"_id": chunk_id},
            {"$set": {"embedding": vector}},
        )
        embedded += 1

    return embedded, errors
