"""
Conversation history router.

Thin, same rule as every other router: parse, delegate to
conversation_service, shape the response. No SQL query logic here.
"""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.models.responses import ConversationEntry, ConversationHistoryResponse
from app.services import conversation_service

logger = get_logger(__name__)
router = APIRouter(tags=["conversations"])


@router.get("/conversations/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str) -> ConversationHistoryResponse:
    records = await conversation_service.get_conversation_history(session_id)

    entries = [
        ConversationEntry(
            question=record["question"],
            answer=record["answer"],
            document_id=record["document_id"],
            sources=record.get("sources", []),
            timestamp=record["timestamp"],
            model=record["model"],
            latency_ms=record["latency_ms"],
            tokens_used=record.get("tokens_used"),
        )
        for record in records
    ]

    logger.info("Retrieved %d conversation entries for session_id=%s", len(entries), session_id)
    return ConversationHistoryResponse(session_id=session_id, conversations=entries)
