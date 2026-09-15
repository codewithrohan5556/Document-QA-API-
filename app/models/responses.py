"""
Pydantic response models shared across routers.

Keeping response shapes here (rather than defining dicts inline in route
handlers) gives FastAPI accurate OpenAPI docs and gives us one place to
change a response shape later without hunting through routers.
"""

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str


class ConversationEntry(BaseModel):
    """
    The HTTP-facing shape of one conversation turn — deliberately narrower
    than ConversationRecord (app/models/database.py). retrieved_chunk_ids
    and raw similarity_scores are internal debugging fields for us, not
    something a client needs; `sources` (the formatted citations) is what
    actually belongs in a response.
    """

    question: str
    answer: str
    document_id: str
    sources: list[str]
    timestamp: datetime
    model: str
    latency_ms: float
    tokens_used: int | None = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    conversations: list[ConversationEntry]


class DeleteResponse(BaseModel):
    document_id: str
    chunks_deleted: int
    status: str


class StatsResponse(BaseModel):
    total_queries: int
    total_conversations: int
    total_documents: int
