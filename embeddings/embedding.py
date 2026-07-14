from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(get_embedding_model_name())
    dimension = model.get_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Embedding model must produce {EMBEDDING_DIMENSION} dimensions, got {dimension}. "
            f"Update the Atlas vector index or choose {DEFAULT_EMBEDDING_MODEL}."
        )
    return model


def get_embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = _load_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def embed_query(question: str) -> list[float]:
    trimmed = question.strip()
    if not trimmed:
        raise ValueError("Question must not be empty")

    vectors = embed_texts([trimmed])
    return vectors[0]
