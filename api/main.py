import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from db.mongo import close_mongo, init_mongo
from pipeline.embed import run_embed
from pipeline.ingest import run_ingest
from pipeline.rag import LlmConfig, run_rag_query, run_rag_query_stream
from scraper.registry import list_sources

load_dotenv()

logger = logging.getLogger(__name__)

_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-8a86fb.log"


def _agent_log(location: str, message: str, data: dict, hypothesis_id: str, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": "8a86fb",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
            "source": "ai-service",
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    except OSError:
        pass


async def _warmup_embedding_model() -> None:
    try:
        from embeddings.embedding import embed_query

        await asyncio.to_thread(embed_query, "warmup")
        logger.info("Embedding model warmed up")
    except Exception as exc:
        logger.warning("Embedding model warmup skipped: %s", exc)


class IngestRequest(BaseModel):
    sources: Optional[list[str]] = Field(
        default=None,
        description="Source names to scrape. Omit to run all configured sources.",
    )
    limit: int = Field(default=50, ge=1, le=200)
    dry_run: bool = Field(
        default=False,
        description="Scrape and chunk without writing to MongoDB.",
    )


class IngestResponse(BaseModel):
    articlesIngested: int
    chunksCreated: int
    sources: list[str]
    errors: list[str]
    dryRun: bool = False


class EmbedRequest(BaseModel):
    batchSize: int = Field(default=64, ge=1, le=500)


class EmbedResponse(BaseModel):
    chunksEmbedded: int
    chunksSkipped: int
    errors: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    model: str = Field(
        min_length=1,
        description="Gemini model from ChatModule (e.g. gemini-2.0-flash).",
    )
    apiKey: str = Field(
        min_length=1,
        description="Decrypted Gemini API key from backend ChatModule.",
    )
    provider: Optional[Literal["gemini", "openrouter"]] = Field(
        default="gemini",
        description="LLM provider from ChatModule. OpenRouter keys (sk-or-…) are auto-detected.",
    )
    history: list[dict[str, str]] = Field(
        default=[],
        description="Conversation history as [{role: 'user'|'assistant', content: str}, ...].",
    )


class SourceResponse(BaseModel):
    source: str
    title: str
    url: str
    publishedAt: str
    similarityScore: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: str
    sources: list[str]


@asynccontextmanager
async def lifespan(_: FastAPI):
    _agent_log(
        "main.py:lifespan:start",
        "ai-service starting",
        {
            "mongoConfigured": bool(os.getenv("MONGODB_URI")),
            "debugLog": str(_DEBUG_LOG),
        },
        "A",
    )
    if os.getenv("MONGODB_URI"):
        await init_mongo()
    await _warmup_embedding_model()
    _agent_log("main.py:lifespan:ready", "ai-service ready", {}, "A")
    yield
    await close_mongo()


app = FastAPI(
    title="News RAG AI Service",
    description="Scrape, embed, retrieve, and generate grounded news answers.",
    lifespan=lifespan,
)


@app.middleware("http")
async def debug_rag_middleware(request: Request, call_next):
    if request.url.path == "/rag/query":
        _agent_log(
            "main.py:middleware:entry",
            "POST /rag/query received",
            {"client": request.client.host if request.client else "unknown"},
            "A",
        )
    try:
        response = await call_next(request)
    except Exception as exc:
        if request.url.path == "/rag/query":
            _agent_log(
                "main.py:middleware:exception",
                "unhandled exception before response",
                {"errorType": type(exc).__name__, "error": str(exc)},
                "A",
            )
        raise

    if request.url.path == "/rag/query":
        _agent_log(
            "main.py:middleware:exit",
            "POST /rag/query response",
            {"statusCode": response.status_code},
            "A",
        )
    return response


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", sources=list_sources())


@app.get("/rag/sources")
async def rag_sources() -> dict[str, list[str]]:
    return {"sources": list_sources()}


@app.post("/rag/ingest", response_model=IngestResponse)
async def rag_ingest(body: IngestRequest) -> IngestResponse:
    if not body.dry_run and not os.getenv("MONGODB_URI"):
        raise HTTPException(
            status_code=503,
            detail="MONGODB_URI is not configured. Set it in ai-service/.env or use dry_run=true.",
        )

    try:
        result = await run_ingest(
            sources=body.sources,
            limit=body.limit,
            dry_run=body.dry_run,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestResponse(**result)


@app.post("/rag/embed", response_model=EmbedResponse)
async def rag_embed(body: EmbedRequest) -> EmbedResponse:
    if not os.getenv("MONGODB_URI"):
        raise HTTPException(
            status_code=503,
            detail="MONGODB_URI is not configured. Set it in ai-service/.env.",
        )

    try:
        result = await run_embed(batch_size=body.batchSize)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EmbedResponse(**result)


@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(body: QueryRequest) -> QueryResponse:
    if not os.getenv("MONGODB_URI"):
        raise HTTPException(
            status_code=503,
            detail="MONGODB_URI is not configured. Set it in ai-service/.env.",
        )

    llm = LlmConfig(
        provider=body.provider or "gemini",
        model=body.model,
        api_key=body.apiKey,
    )

    _agent_log(
        "main.py:rag_query:entry",
        "rag/query request received",
        {
            "provider": llm.provider,
            "model": llm.model,
            "questionLen": len(body.question),
            "apiKeyLen": len(body.apiKey),
        },
        "E",
    )

    try:
        result = await run_rag_query(body.question, llm=llm, history=body.history)
        _agent_log(
            "main.py:rag_query:success",
            "rag/query completed",
            {
                "sourceCount": len(result.get("sources") or []),
                "answerLen": len(result.get("answer") or ""),
            },
            "A",
        )
        return QueryResponse(**result)
    except ValueError as exc:
        _agent_log(
            "main.py:rag_query:value_error",
            "rag/query validation error",
            {"error": str(exc)},
            "D",
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        _agent_log(
            "main.py:rag_query:runtime_error",
            "rag/query runtime error",
            {"error": str(exc)},
            "D",
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        _agent_log(
            "main.py:rag_query:exception",
            "rag/query unhandled exception",
            {"errorType": type(exc).__name__, "error": str(exc)},
            "A",
        )
        logger.exception("rag/query failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/rag/query/stream")
async def rag_query_stream(body: QueryRequest) -> StreamingResponse:
    if not os.getenv("MONGODB_URI"):
        raise HTTPException(
            status_code=503,
            detail="MONGODB_URI is not configured. Set it in ai-service/.env.",
        )

    llm = LlmConfig(
        provider=body.provider or "gemini",
        model=body.model,
        api_key=body.apiKey,
    )

    async def event_generator():
        try:
            async for event in run_rag_query_stream(body.question, llm=llm, history=body.history):
                yield f"data: {json.dumps(event)}\n\n"
        except ValueError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        except Exception as exc:
            logger.exception("rag/query/stream failed")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
