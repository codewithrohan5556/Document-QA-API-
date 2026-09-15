"""
Models mirroring what actually gets stored in PostgreSQL (migrated from
MongoDB — see app/db/postgres.py for the corresponding CREATE TABLE) —
distinct from requests.py/responses.py, which describe the HTTP contract.
Only conversation_service.py reads/writes these.

Field types below are unchanged from the MongoDB version: list[str],
list[int], list[float] map directly onto Postgres's native array types
(TEXT[], INTEGER[], DOUBLE PRECISION[]) via asyncpg, with no serialization
code needed on either side.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ConversationRecord(BaseModel):
    # Groups queries into a thread — what GET /conversations/{session_id}
    # (Step 12) will filter on.
    session_id: str

    # Which document the question was about — lets history be scoped/filtered
    # per document later if needed.
    document_id: str

    # The raw question, for display and audit.
    question: str

    # The generated answer text (sources stripped back off — see
    # rag_service.stream_and_record_answer) — stored so history can be
    # displayed without re-running the LLM.
    answer: str

    # Formatted citations ("file.pdf, page 3") shown alongside the answer —
    # cheap to store, expensive to recompute from raw chunk metadata later.
    sources: list[str] = Field(default_factory=list)

    # Which chunks were retrieved and how relevant each was. NOT the chunk
    # text itself — that's already durably stored in the vector database,
    # so duplicating it here would be unnecessary data per the spec's
    # "do not store unnecessary data" rule. These are for debugging poor
    # answers: "was a low-relevance chunk retrieved?"
    retrieved_chunk_ids: list[int] = Field(default_factory=list)
    similarity_scores: list[float] = Field(default_factory=list)

    # Which LLM produced the answer — useful once the model changes over time
    model: str

    # Cost/usage tracking. Optional because streaming responses (our
    # /query path) don't reliably expose token counts the way a single
    # non-streaming completion does — left as a documented gap, not silently
    # faked with a wrong number.
    tokens_used: int | None = None

    # Performance tracking — how long the full answer took to generate.
    latency_ms: float

    # When this happened — for chronological ordering in conversation history.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
