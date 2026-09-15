"""Pydantic request models for incoming API payloads."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    document_id: str
    question: str = Field(..., min_length=1)
    session_id: str | None = None
