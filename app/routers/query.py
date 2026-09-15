"""
Query router.

Deliberately thin, same rule as documents.py: parse the request, delegate,
shape the response. No context-building or LLM logic lives here — that's
rag_service.py. The one thing worth noting is the ORDER of calls below; see
the comment inline and rag_service.stream_answer()'s docstring for why.
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.models.requests import QueryRequest
from app.services import rag_service
from app.services.retrieval_service import retrieve_chunks

logger = get_logger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query")
async def query_document(payload: QueryRequest) -> StreamingResponse:
    # Retrieval happens HERE, before the StreamingResponse is constructed —
    # not inside rag_service.stream_answer(). If embedding the question or
    # searching the vector store fails, it raises VectorStoreError right
    # here, which the global exception handler turns into a clean JSON
    # error response. Once StreamingResponse starts sending, HTTP headers
    # (and the 200 status) are already committed — there's no clean way to
    # turn a mid-stream failure into a 502 anymore.
    chunks = retrieve_chunks(payload.question, document_id=payload.document_id)

    logger.info(
        "Streaming answer: document_id=%s session_id=%s chunks=%d",
        payload.document_id, payload.session_id, len(chunks),
    )

    return StreamingResponse(
        rag_service.stream_and_record_answer(
            payload.question, chunks, payload.document_id, payload.session_id
        ),
        media_type="text/plain",
    )
