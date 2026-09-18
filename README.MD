# Document Q&A API

A production-style **Retrieval-Augmented Generation (RAG)** backend that lets users upload documents, ask questions about them and receive grounded, cited answers — streamed in real time.

---

## Features

- **Upload documents** — PDF and `.txt` files, with page-aware text extraction
- **Semantic search** — embeddings stored in ChromaDB (local) or Pinecone (cloud)
- **Streaming answers** — tokens arrive in real time, not as one buffered blob
- **Source citations** — every answer cites the exact page or chunk it came from
- **Conversation history** — every query persisted to PostgreSQL with latency, model, and sources
- **LangSmith observability** — every LLM call, retrieval operation, and context build is traced
- **Swappable vector store** — switch between ChromaDB and Pinecone with one env var
- **Docker support** — full `docker-compose.yml` for local containerized deployment
- **71 tests** — unit, integration, and end-to-end smoke test

---

## Architecture

<p align="center">  <img src="images/Architecture.png" alt="" width="100%"></p>

## User Interface
<p align="center">  <img src="images/UI1.png" alt="" width="100%"></p>
<p align="center">  <img src="images/UI2.png" alt="" width="100%"></p>


### Request pipeline

```
POST /query
    │
    ├── Embed question (NVIDIA via OpenRouter)
    ├── Vector search (ChromaDB or Pinecone)
    ├── Retrieve top-k chunks
    ├── Build grounded prompt
    ├── Stream answer token by token (Groq)
    ├── Append source citations
    └── Persist to PostgreSQL
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Pydantic |
| RAG | LangChain, LangSmith |
| LLM | Groq (`openai/gpt-oss-120b`) via OpenAI-compatible API |
| Embeddings | NVIDIA `llama-nemotron-embed-vl-1b-v2` via OpenRouter |
| Vector DB (Phase 1) | ChromaDB (local, embedded) |
| Vector DB (Phase 2) | Pinecone (managed, serverless) |
| Database | PostgreSQL via asyncpg |
| Observability | LangSmith |
| Containerization | Docker, Docker Compose |
| Testing | pytest, pytest-asyncio, httpx |
| Manual testing UI | Streamlit |

---

## Project Structure

```
document-qa-api/
│
├── app/
│   ├── main.py                    # FastAPI app, lifespan, exception handlers
│   │
│   ├── routers/
│   │   ├── documents.py           # POST /upload, DELETE /document/{id}, GET /stats
│   │   ├── query.py               # POST /query (streaming)
│   │   └── conversations.py       # GET /conversations/{session_id}
│   │
│   ├── models/
│   │   ├── requests.py          # QueryRequest
│   │   ├── responses.py         # UploadResponse,ConversationEntry,StatsResponse,   
│   │   └── database.py          # ConversationRecord (PostgreSQL schema mirror)
│   │
│   ├── services/
│   │   ├── document_service.py    # File validation, page-aware text extraction
│   │   ├── ingestion_service.py   # Chunking, embedding, indexing
│   │   ├── retrieval_service.py   # Question embedding + vector search
│   │   ├── rag_service.py         # Prompt build, LLM call, streaming, persistence
│   │   ├── llm_service.py         # Groq ChatOpenAI singleton
│   │   ├── embedding_service.py   # NVIDIA OpenAIEmbeddings singleton
│   │   └── conversation_service.py# PostgreSQL read/write
│   │
│   ├── vectorstores/
│   │   ├── base.py                # VectorStore abstract interface
│   │   ├── chroma.py              # ChromaDB implementation
│   │   ├── pinecone.py            # Pinecone implementation
│   │   └── __init__.py            # Factory: get_vector_store()
│   │
│   ├── db/
│   │   └── postgres.py            # asyncpg pool lifecycle, schema creation
│   │
│   ├── core/
│   │   ├── config.py              # Pydantic Settings — all env vars in one place
│   │   ├── logging.py             # Structured logging setup
│   │   ├── prompts.py             # RAG prompt template
│   │   └── tracing.py             # LangSmith setup — bridges Settings to os.environ
│   │
│   └── utils/
│       ├── exceptions.py          # Custom exceptions → clean JSON error responses
│       └── helpers.py             # generate_document_id()
│
├── tests/
│
├── tools/
│   ├── streamlit_app.py           # Manual testing UI (separate venv)
│   ├── requirements.txt           # streamlit, requests only
│   └── README.md
│
├── data/                          # ChromaDB persists here (gitignored)
├── .env.example                   # All env vars documented
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Installation

### Prerequisites

- Python 3.12
- PostgreSQL (local or managed)
- API keys: [Groq](https://console.groq.com), [OpenRouter](https://openrouter.ai), [Pinecone](https://app.pinecone.io), [LangSmith](https://smith.langchain.com)

### Local setup

```bash
git clone https://github.com/YOUR_USERNAME/document-qa-api.git
cd document-qa-api

python3.12 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys — see Environment Variables section below
```

---

## Environment Variables

All configuration lives in `.env`. Copy `.env.example` and fill in your values.

```env
# LLM (Groq — OpenAI-compatible)
GROQ_API_KEY=

# Embeddings (NVIDIA via OpenRouter — OpenAI-compatible)
OPENROUTER_API_KEY=

# Vector store: "chroma" (local) or "pinecone" (cloud)
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma

# Pinecone (only needed when VECTOR_STORE=pinecone)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=document-qa
PINECONE_DIMENSION=2048

# PostgreSQL
POSTGRES_DSN=postgresql://user:password@localhost:5432/document_qa

# LangSmith observability
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=advanced-rag
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# RAG behavior
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=5
```

---

## Running Locally

```bash
# Start the API
uvicorn app.main:app --reload --port 8000
```

Expected startup output:
```
INFO: Starting Document Q&A API (vector_store=pinecone)
INFO: LangSmith tracing enabled (project=advanced-rag)
INFO: Connected to PostgreSQL and ensured schema exists
INFO: PineconeVectorStore ready (index=document-qa, dimension=2048)
INFO: Uvicorn running on http://127.0.0.1:8000
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Manual testing UI (Streamlit)

Run in a **separate terminal with its own venv** — never install Streamlit into the project venv as it upgrades Starlette and breaks the API:

```bash
python3 -m venv ~/.venv-streamlit
~/.venv-streamlit/bin/pip install -r tools/requirements.txt
~/.venv-streamlit/bin/streamlit run tools/streamlit_app.py
```

Open `http://localhost:8501`.

---

## Docker

```bash
# Copy and fill in your keys
cp .env.example .env

# Build and start everything (API + PostgreSQL)
docker compose up --build
```

The API runs on `http://localhost:8000`. PostgreSQL data persists in a named Docker volume.

> **Note:** Docker uses `VECTOR_STORE=pinecone` by default since ChromaDB's local storage doesn't survive container rebuilds. Pinecone and PostgreSQL are both cloud/managed services, so no data is lost on redeploy.

---

## API Endpoints

### Upload a document
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF or .txt file>
```
```json
{
  "document_id": "abc123def456",
  "filename": "research_paper.pdf",
  "chunks_created": 42,
  "status": "indexed"
}
```

### Ask a question (streaming)
```http
POST /query
Content-Type: application/json

{
  "document_id": "abc123def456",
  "question": "What are the main findings?",
  "session_id": "my-session-1"
}
```
Response streams token by token, ending with:
```
...the main findings include X, Y, and Z [page 3].

Sources:
- research_paper.pdf, page 3
- research_paper.pdf, page 7
```

### Get conversation history
```http
GET /conversations/{session_id}
```
```json
{
  "session_id": "my-session-1",
  "conversations": [
    {
      "question": "What are the main findings?",
      "answer": "The main findings include...",
      "sources": ["research_paper.pdf, page 3"],
      "model": "openai/gpt-oss-120b",
      "latency_ms": 2341.5,
      "timestamp": "2026-08-25T10:30:00Z"
    }
  ]
}
```

### Stats
```http
GET /stats
```
```json
{
  "total_documents": 5,
  "total_queries": 23,
  "total_conversations": 8
}
```

### Delete a document
```http
DELETE /document/{document_id}
```
```json
{
  "document_id": "abc123def456",
  "chunks_deleted": 42,
  "status": "deleted"
}
```

### Health check
```http
GET /health
```
```json
{"status": "healthy"}
```

---

## RAG Pipeline

### Document ingestion

```
Upload
  ↓
File validation (.pdf / .txt only)
  ↓
Page-aware text extraction (pypdf)
  ↓
Chunk per page (1000 chars, 200 overlap)
  ↓
Embed each chunk (NVIDIA via OpenRouter, 2048 dims)
  ↓
Store vectors + metadata (Pinecone)
```

Metadata stored per chunk: `document_id`, `filename`, `page_number`, `chunk_id`, `source`, `text`.
Page numbers are preserved so citations say `"file.pdf, page 3"` not `"file.pdf, chunk 12"`.

### Query pipeline

```
Question
  ↓
Embed question (same model as ingestion)
  ↓
Vector similarity search (cosine, top-k)
  ↓
Retrieve chunks + metadata
  ↓
Build grounded prompt (context + rules: no hallucination)
  ↓
Stream answer token by token (Groq)
  ↓
Append source citations
  ↓
Persist to PostgreSQL
```

### RAG prompt rules

The LLM is explicitly instructed to:
- Answer using **only** the supplied context
- Never invent information not in the document
- Say so clearly if the answer isn't in the context
- Reference source chunks using `[page N]` labels

---

## Pinecone

<p align="center">  <img src="images/Pinecone1.png" alt="" width="100%"></p>
<p align="center">  <img src="images/Pinecone2.png" alt="" width="100%"></p>
---

## PostgreSQL Schema

```sql
CREATE TABLE conversations (
    id                  SERIAL PRIMARY KEY,
    session_id          TEXT NOT NULL,
    document_id         TEXT NOT NULL,
    question            TEXT NOT NULL,
    answer              TEXT NOT NULL,
    sources             TEXT[] NOT NULL DEFAULT '{}',
    retrieved_chunk_ids INTEGER[] NOT NULL DEFAULT '{}',
    similarity_scores   DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    model               TEXT NOT NULL,
    tokens_used         INTEGER,
    latency_ms          DOUBLE PRECISION NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_conversations_session_id ON conversations (session_id);
```

The schema is created automatically on first startup — no manual migration needed.

---

## LangSmith Observability
<p align="center">  <img src="images/Langsmith1.png" alt="" width="100%"></p>
<p align="center">  <img src="images/Langsmith2.png" alt="" width="100%"></p>
<p align="center">  <img src="images/Langsmith3.png" alt="" width="100%"></p>


Every `/query` call produces traces visible in [smith.langchain.com](https://smith.langchain.com):

| Trace | Type | What it shows |
|---|---|---|
| `retrieve_chunks` | retriever | Question, retrieved chunks, similarity scores, latency |
| `build_context` | prompt | Exact context string sent to LLM |
| `stream_answer` | chain | LLM call with full prompt |
| `ChatOpenAI` | llm | Auto-traced by LangChain — tokens, latency |
| `stream_and_record_answer` | chain | Complete assembled answer, sources, total latency |

### Debugging poor answers

1. **Open `retrieve_chunks`** — were relevant chunks retrieved? If not, the problem is retrieval (chunking/embeddings), not the LLM
2. **Open `build_context`** — does the context contain what the LLM needed? If not, wrong chunks were retrieved
3. **Open `ChatOpenAI`** — given the right context, did the LLM still answer badly? If yes, edit the prompt in `app/core/prompts.py`

---

## Testing

```bash
pytest tests/ -v
```

**71 tests** covering:

| Test file | What it covers |
|---|---|
| `test_health.py` | `/health` endpoint |
| `test_documents.py` | Upload, delete, stats — success and error cases |
| `test_query.py` | Streaming, validation, persistence wiring, retrieval failure handling |
| `test_retrieval.py` | Chunk retrieval, top-k, document-id filtering |
| `test_rag_service.py` | Answer generation, no-context short-circuit, LLM failure |
| `test_citations.py` | PDF page extraction, page-number citations end to end |
| `test_conversation_service.py` | PostgreSQL INSERT, count, graceful degradation |
| `test_conversations.py` | History retrieval, empty session, 503 on Postgres down |
| `test_vectorstores.py` | Factory switch (Chroma → Pinecone), unknown backend error |
| `test_pinecone_vectorstore.py` | All Pinecone methods mocked — filter translation, batching, shape normalization |
| `test_ingestion_service.py` | Chunking edge cases, validation guards |
| `test_error_handling.py` | 500 handler, 404/405, OpenAPI schema validity |
| `test_end_to_end_smoke.py` | Full pipeline: upload → query → stream → persist → history → stats → delete |

External services (Groq, OpenRouter, Pinecone, PostgreSQL) are mocked in tests. ChromaDB runs real (local embedded, no server needed).

---

## Deployment on Render

1. Push to GitHub
2. Create a **PostgreSQL** database on Render (free tier) — copy the Internal Database URL
3. Create a **Web Service** connected to your repo:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add all environment variables from `.env.example` in the Render dashboard
5. Set `POSTGRES_DSN` to the Internal Database URL from step 2
6. Set `VECTOR_STORE=pinecone` (Pinecone survives redeploys; ChromaDB disk storage does not)

Every `git push origin main` triggers an automatic redeploy.

---

## Known Limitations

- **Scanned PDFs** — documents with no text layer (image-only PDFs) return a 400 error. OCR is not implemented.
- **Token counts** — streaming responses (`/query`) don't reliably surface token usage from the provider. `tokens_used` is stored as `null` in PostgreSQL.
- **LangSmith trace nesting** — retrieval and generation appear as two separate top-level traces per query rather than one unified tree. This is a deliberate architectural trade-off (retrieval runs before streaming starts, for clean error handling).
- **Pinecone `count_documents()`** — uses a full metadata scan to count distinct document IDs. Fine for this project's scale; would need a separate registry for a large corpus.

---

## Future Improvements

- OCR support for scanned PDFs (via Tesseract or a cloud OCR API)
- Hybrid search (BM25 + semantic similarity)
- Re-ranking with a cross-encoder
- API key authentication for multi-user deployments
- Streaming token count estimation
- Unified LangSmith trace tree (retrieval + generation in one span)
- Support for more file types (DOCX, CSV, web URLs)

---
