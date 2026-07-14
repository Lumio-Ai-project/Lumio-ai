# FastAPI RAG pipeline

## Phase 4 — Data layer

```
scraper/  →  preprocessing/  →  MongoDB
   │              │
   │         cleaner.py   strip HTML, remove boilerplate
   │         chunker.py   paragraph-based semantic chunks
   │
   ├── bbc.py        BBC articles from bbclinks.txt
   └── rss.py        RSS feeds (TechCrunch, Verge, etc.)
```

## Phase 5 — RAG core

```
embeddings/  →  chunks.embedding in MongoDB
retrieval/   →  Atlas Vector Search on chunks
llm/         →  Gemini generation
pipeline/    →  embed + rag orchestration
```

### End-to-end workflow

```bash
uvicorn api.main:app --reload --port 8000

# 1. Ingest articles and text chunks (Phase 4)
POST /rag/ingest

# 2. Generate embeddings for chunks without an embedding field
POST /rag/embed

# 3. Ask a grounded question
POST /rag/query
```

## Setup

```bash
cd ai-service
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Configure [`ai-service/.env`](.env) (same MongoDB as backend):

| Variable | Required | Purpose |
| -------- | -------- | ------- |
| `MONGODB_URI` | Yes | MongoDB Atlas connection string |
| `MONGODB_DB` | Recommended | Database name (e.g. `Lumio`) |
| `EMBEDDING_MODEL` | Optional | Default `sentence-transformers/all-MiniLM-L6-v2` |
| `RAG_TOP_K` | Optional | Retrieved chunks per query (default `5`) |
| `VECTOR_SEARCH_INDEX` | Optional | Atlas index name (default `chunks_vector_index`) |

Gemini **API keys and models are not stored in ai-service**. The backend decrypts them from **ChatModule** (`provider`, `model`, `apiKeyEncrypted`) and forwards `provider`, `model`, and `apiKey` on every `POST /rag/query` call.

### Atlas Vector Search index

Create this index manually in the MongoDB Atlas UI before running `/rag/query`:

- **Database:** same as `MONGODB_DB`
- **Collection:** `chunks`
- **Index name:** `chunks_vector_index` (or set `VECTOR_SEARCH_INDEX`)
- **Type:** Vector Search
- **Field:** `embedding`
- **Dimensions:** `384`
- **Similarity:** `cosine`

Optional filter fields: `metadata.source`, `metadata.category`.

Example index definition (Atlas Search JSON editor):

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

## Run ingest

**CLI** (scrape → clean → chunk → store):

```bash
python -m scraper.run --source bbc --limit 20
python -m scraper.run --source techcrunch --source sciencedaily --limit 10
python -m scraper.run --dry-run --source bbc --limit 5   # no MongoDB write
```

**API**:

```http
POST /rag/ingest
Content-Type: application/json

{
  "sources": ["bbc", "techcrunch"],
  "limit": 20
}
```

## Run embed

Generates Sentence Transformer embeddings for chunks that do not yet have an `embedding` field. Safe to run repeatedly.

```http
POST /rag/embed
Content-Type: application/json

{
  "batchSize": 64
}
```

Response:

```json
{
  "chunksEmbedded": 120,
  "chunksSkipped": 0,
  "errors": []
}
```

## Run query

The backend loads the selected **ChatModule**, decrypts `apiKeyEncrypted`, and calls this endpoint with `provider`, `model`, and `apiKey`. Direct calls (e.g. FastAPI `/docs`) must include the same fields:

```http
POST /rag/query
Content-Type: application/json

{
  "question": "What happened in AI today?",
  "provider": "gemini",
  "model": "gemini-2.0-flash",
  "apiKey": "decrypted-key-from-chat-module"
}
```

Response:

```json
{
  "answer": "Based on retrieved articles...",
  "sources": [
    {
      "source": "Reuters",
      "title": "Article title",
      "url": "https://www.reuters.com/...",
      "publishedAt": "2026-06-26T10:00:00Z",
      "similarityScore": 0.94
    }
  ]
}
```

### Error responses

| Status | When |
| ------ | ---- |
| 400 | Empty question, missing `model`/`apiKey`, invalid batch size, unsupported provider |
| 502 | Gemini API failure |
| 503 | Missing `MONGODB_URI` |

## Sources

| Key | Type | Category |
| --- | ---- | -------- |
| `bbc` | HTML (bbclinks.txt) | general |
| `techcrunch` | RSS | technology |
| `theverge` | RSS | technology |
| `tnw` | RSS | technology |
| `venturebeat` | RSS | technology |
| `sciencedaily` | RSS | science |
| `physorg` | RSS | science |
| `openai` | RSS | ai |
| `huggingface` | RSS | ai |
| `reuters` | RSS | general |
| `apnews` | RSS | general |

## MongoDB collections

- `news_articles` — deduplicated by `url`
- `chunks` — linked via `articleId` + `chunkIndex`, with optional `embedding` vector (384 dims)

## Manual verification

After ingest and embed, try questions from [`docs/example-questions.md`](../docs/example-questions.md) via FastAPI `/docs` or curl. Expect grounded answers with at least one source URL when relevant chunks exist.
