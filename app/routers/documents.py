"""
Document router.

Deliberately thin: parses requests, delegates to services, shapes
responses. No validation logic, no text extraction, no chunking logic, no
embedding/storage logic, no vector-store internals live here — that's
document_service.py / ingestion_service.py / vectorstores/.

Pipeline: validate -> extract pages -> chunk (page-aware) -> embed -> store -> respond.
"""

from fastapi import APIRouter, File, UploadFile

from app.core.logging import get_logger
from app.models.responses import DeleteResponse, StatsResponse, UploadResponse
from app.services import conversation_service, document_service, ingestion_service
from app.utils.exceptions import DocumentNotFoundError, EmptyDocumentError
from app.utils.helpers import generate_document_id
from app.vectorstores import get_vector_store

logger = get_logger(__name__)
router = APIRouter(tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    ext = document_service.validate_file(file)

    raw_bytes = await file.read()
    if not raw_bytes:
        raise EmptyDocumentError("Uploaded file is empty.")

    pages = document_service.extract_pages(raw_bytes, ext)
    chunks = ingestion_service.chunk_pages(pages, ext)

    document_id = generate_document_id()
    ingestion_service.embed_and_index_chunks(document_id, file.filename, chunks)

    logger.info(
        "Ingested document_id=%s filename=%s pages=%d chunks_created=%d",
        document_id, file.filename, len(pages), len(chunks),
    )

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        chunks_created=len(chunks),
        status="indexed",
    )


@router.delete("/document/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str) -> DeleteResponse:
    """
    Removes every chunk belonging to document_id from the active vector
    store. Works against whichever backend VECTOR_STORE points at — this
    router never knows or cares whether that's Chroma or (Phase 2) Pinecone.

    Existence check comes for free from the delete count: 0 chunks deleted
    means the document_id never existed (or was already deleted), so we
    raise 404 rather than silently returning success either way.
    """
    vector_store = get_vector_store()
    deleted_count = vector_store.delete(document_id)

    if deleted_count == 0:
        raise DocumentNotFoundError(f"No document found with document_id '{document_id}'.")

    logger.info("Deleted document_id=%s (%d chunks removed)", document_id, deleted_count)
    return DeleteResponse(document_id=document_id, chunks_deleted=deleted_count, status="deleted")


@router.get("/stats", response_model=StatsResponse)
async def get_stats() -> StatsResponse:
    """
    Aggregate counts across both storage backends: document count from the
    vector store, query/session counts from PostgreSQL (degrading to 0 rather
    than erroring if Postgres isn't configured — see
    conversation_service.count_queries()'s docstring for why that's a
    different choice than GET /conversations/{session_id} makes).
    """
    vector_store = get_vector_store()
    total_documents = vector_store.count_documents()
    total_queries = await conversation_service.count_queries()
    total_conversations = await conversation_service.count_distinct_sessions()

    return StatsResponse(
        total_queries=total_queries,
        total_conversations=total_conversations,
        total_documents=total_documents,
    )
