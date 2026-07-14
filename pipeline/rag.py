from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from embeddings.embedding import embed_query
from llm.client import GeminiError, generate_answer, generate_answer_stream, resolve_llm_config
from pipeline.history import (
    HistoryMessage,
    should_retrieve_news,
    rewrite_query_with_history,
    summarize_history,
)
from pipeline.query_preprocessing import build_metadata_filter, preprocess_query
from prompt import (
    CONVERSATIONAL_SYSTEM_PROMPT,
    NEWS_SYSTEM_PROMPT,
    build_conversational_user_prompt,
    build_history_messages,
    build_rag_user_prompt,
)
from retrieval.ensemble import search_ensemble_chunks
from retrieval.vector_search import RetrievedChunk, get_default_top_k


@dataclass(frozen=True)
class LlmConfig:
    model: str
    api_key: str
    provider: str = "gemini"


@dataclass(frozen=True)
class RagSource:
    source: str
    title: str
    url: str
    published_at: str
    similarity_score: float


@dataclass(frozen=True)
class PreparedRagPrompt:
    system_prompt: str
    user_prompt: str
    retrieved: list[RetrievedChunk]
    processed_history: list[HistoryMessage]
    include_sources: bool


async def _prepare_rag_prompt(
    question: str,
    llm: LlmConfig,
    history: list[HistoryMessage] | None = None,
) -> PreparedRagPrompt:
    trimmed = question.strip()
    if not trimmed:
        raise ValueError("Question must not be empty")

    resolve_llm_config(
        provider=llm.provider,
        model=llm.model,
        api_key=llm.api_key,
    )

    processed_history = summarize_history(history or [], llm=llm)
    use_news_retrieval = should_retrieve_news(trimmed, processed_history)

    if not use_news_retrieval:
        return PreparedRagPrompt(
            system_prompt=CONVERSATIONAL_SYSTEM_PROMPT,
            user_prompt=build_conversational_user_prompt(trimmed),
            retrieved=[],
            processed_history=processed_history,
            include_sources=False,
        )

    retrieval_query = await asyncio.to_thread(
        rewrite_query_with_history,
        trimmed,
        processed_history,
        llm=llm,
    )

    preprocessed = preprocess_query(retrieval_query, llm=None)
    metadata_filter = build_metadata_filter(preprocessed.filters)
    retrieval_query = preprocessed.rewritten_query or retrieval_query

    query_embedding = await asyncio.to_thread(embed_query, retrieval_query)
    retrieved = await search_ensemble_chunks(
        query_embedding,
        retrieval_query,
        top_k=get_default_top_k(),
        metadata_filter=metadata_filter,
    )

    retrieved = _filter_relevant_chunks(retrieved)
    context = _build_context(retrieved) if retrieved else ""
    user_prompt = build_rag_user_prompt(trimmed, context, history=processed_history or None)
    return PreparedRagPrompt(
        system_prompt=NEWS_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        retrieved=retrieved,
        processed_history=processed_history,
        include_sources=bool(retrieved),
    )


async def run_rag_query(
    question: str,
    llm: LlmConfig,
    history: list[HistoryMessage] | None = None,
) -> dict[str, Any]:
    prepared = await _prepare_rag_prompt(question, llm, history)

    history_messages = (
        build_history_messages(prepared.processed_history)
        if prepared.processed_history
        else None
    )

    try:
        answer = await asyncio.to_thread(
            generate_answer,
            prepared.system_prompt,
            prepared.user_prompt,
            provider=llm.provider,
            model=llm.model,
            api_key=llm.api_key,
            history=history_messages,
        )
    except GeminiError as exc:
        raise RuntimeError(str(exc)) from exc

    clean_answer, grounded = _parse_grounding_marker(answer)
    sources = _build_sources(prepared) if grounded else []
    return {
        "answer": clean_answer,
        "sources": sources,
    }


async def run_rag_query_stream(
    question: str,
    llm: LlmConfig,
    history: list[HistoryMessage] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield SSE-style events: token chunks then a done event with sources."""
    prepared = await _prepare_rag_prompt(question, llm, history)

    history_messages = (
        build_history_messages(prepared.processed_history)
        if prepared.processed_history
        else None
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    # Accumulate full answer to parse grounding marker after streaming
    accumulated: list[str] = []

    def _produce() -> None:
        """Stream tokens to the queue, suppressing the grounding-marker first line."""
        try:
            # Buffer tokens until we have the first complete line (the marker).
            # Once the marker is resolved we flush buffered content (minus the
            # marker line itself) and stream the rest normally.
            first_line_buf: list[str] = []
            first_line_done = False
            marker_stripped_prefix = ""  # text after the marker on the same or next line

            for token in generate_answer_stream(
                prepared.system_prompt,
                prepared.user_prompt,
                provider=llm.provider,
                model=llm.model,
                api_key=llm.api_key,
                history=history_messages,
            ):
                accumulated.append(token)

                if not first_line_done:
                    first_line_buf.append(token)
                    joined = "".join(first_line_buf)
                    # Wait until we see a newline so we have the full first line
                    if "\n" in joined:
                        first_line_done = True
                        first_line, rest = joined.split("\n", 1)
                        first_line_stripped = first_line.lstrip()
                        if first_line_stripped.startswith("[GROUNDED]"):
                            after_marker = first_line_stripped[len("[GROUNDED]"):]
                            rest_to_emit = (after_marker + "\n" + rest).lstrip("\n\r ")
                        elif first_line_stripped.startswith("[NOT_GROUNDED]"):
                            after_marker = first_line_stripped[len("[NOT_GROUNDED]"):]
                            rest_to_emit = (after_marker + "\n" + rest).lstrip("\n\r ")
                        else:
                            # No marker — emit the whole buffered content as-is
                            rest_to_emit = joined
                        if rest_to_emit:
                            loop.call_soon_threadsafe(
                                queue.put_nowait, {"type": "token", "content": rest_to_emit}
                            )
                else:
                    loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "content": token})

            # If the stream ended before we saw a newline (very short answer)
            if not first_line_done and first_line_buf:
                joined = "".join(first_line_buf)
                _, _ = _parse_grounding_marker(joined)  # used below via full_answer
                clean, _ = _parse_grounding_marker(joined)
                if clean:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": "token", "content": clean}
                    )

            full_answer = "".join(accumulated)
            _, grounded = _parse_grounding_marker(full_answer)
            resolved_sources = _build_sources(prepared) if grounded else []
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "sources": resolved_sources})
        except Exception as exc:
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "error", "message": str(exc)},
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    producer = asyncio.create_task(asyncio.to_thread(_produce))

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if item.get("type") == "error":
                raise RuntimeError(str(item.get("message") or "Streaming generation failed"))
            yield item
    finally:
        await producer


def _parse_grounding_marker(answer: str) -> tuple[str, bool]:
    """Strip the [GROUNDED]/[NOT_GROUNDED] marker from the LLM answer.

    Returns (clean_answer, is_grounded). If no marker is found we default
    to grounded=True so existing sources are not accidentally hidden.
    """
    stripped = answer.lstrip()
    if stripped.startswith("[GROUNDED]"):
        clean = stripped[len("[GROUNDED]"):].lstrip("\n\r ")
        return clean, True
    if stripped.startswith("[NOT_GROUNDED]"):
        clean = stripped[len("[NOT_GROUNDED]"):].lstrip("\n\r ")
        return clean, False
    # No marker found — keep answer as-is and preserve sources
    return answer, True


def _get_min_source_score() -> float:
    raw = os.getenv("RAG_MIN_SOURCE_SCORE", "0.018").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.018


def _filter_relevant_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    ranked = sorted(chunks, key=lambda item: item.similarity_score, reverse=True)
    top_score = ranked[0].similarity_score
    if top_score < _get_min_source_score():
        return []

    cutoff = top_score * 0.6
    return [chunk for chunk in ranked if chunk.similarity_score >= cutoff]


def _build_sources(prepared: PreparedRagPrompt) -> list[dict[str, Any]]:
    if not prepared.include_sources:
        return []
    return [_format_source(chunk) for chunk in _dedupe_sources(prepared.retrieved)]


def _build_context(chunks: list[RetrievedChunk]) -> str:
    sections: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        header_parts = [part for part in [chunk.source, chunk.title] if part]
        header = " — ".join(header_parts) if header_parts else f"Chunk {index}"
        sections.append(f"[{index}] {header}\n{chunk.text}")
    return "\n\n".join(sections)


def _dedupe_sources(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    best_by_url: dict[str, RetrievedChunk] = {}

    for chunk in chunks:
        key = chunk.url or chunk.chunk_id
        existing = best_by_url.get(key)
        if existing is None or chunk.similarity_score > existing.similarity_score:
            best_by_url[key] = chunk

    return sorted(best_by_url.values(), key=lambda item: item.similarity_score, reverse=True)


def _format_source(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "source": chunk.source,
        "title": chunk.title,
        "url": chunk.url,
        "publishedAt": chunk.published_at,
        "similarityScore": round(chunk.similarity_score, 4),
    }
