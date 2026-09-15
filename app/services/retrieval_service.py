"""
Retrieval service: turns a question into ranked, cited chunks.

Pipeline: question -> embed -> vector_store.query() -> shape results into
RetrievedChunk objects. Depends only on `get_vector_store()` (the VectorStore
interface) and `get_embeddings()` (the embedding-provider interface) — same
rule as ingestion_service.py. rag_service.py (Step 8) will call
`retrieve_chunks()` and never touch the vector store or embeddings directly.

@traceable (Step 14): marks this as a named "retriever" span in LangSmith,
distinct from the LLM call that LangChain already auto-traces. When tracing
is disabled (the default — see app/core/tracing.py), this decorator is an
inert no-op; it does not change behavior, return values, or performance in
any way you'd notice.
"""

from langsmith import traceable
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import get_embeddings
from app.utils.exceptions import VectorStoreError
from app.vectorstores import get_vector_store

logger = get_logger(__name__)


class RetrievedChunk(BaseModel):
    text: str
    document_id: str
    filename: str
    chunk_id: int
    # -1 sentinel means "no real page concept" (e.g. this chunk came from a
    # .txt file). Positive values are real 1-based PDF page numbers. See
    # ingestion_service.chunk_pages() for where this is assigned.
    page_number: int
    # Cosine similarity approximation (1 - cosine_distance), range roughly
    # [-1, 1] in theory but [0, 1] in practice for embeddings like these.
    # Higher = more relevant. Only meaningful because chroma.py explicitly
    # configures the collection to use cosine distance.
    score: float


@traceable(name="retrieve_chunks", run_type="retriever")
def retrieve_chunks(
    question: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Embeds `question` and returns the top-k most similar chunks.

    `document_id`, when given, scopes the search to a single document via
    metadata filtering (matches the /query request contract: users ask
    questions about one uploaded document at a time). `top_k` falls back to
    settings.top_k when not given — never hardcoded here.
    """
    top_k = top_k if top_k is not None else settings.top_k

    try:
        embeddings_model = get_embeddings()
        query_vector = embeddings_model.embed_query(question)
    except Exception as exc:
        logger.error("Failed to embed question: %s", exc)
        raise VectorStoreError(f"Failed to embed question: {exc}") from exc

    filters = {"document_id": document_id} if document_id else None

    try:
        vector_store = get_vector_store()
        results = vector_store.query(embedding=query_vector, top_k=top_k, filters=filters)
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        raise VectorStoreError(f"Vector search failed: {exc}") from exc

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    chunks: list[RetrievedChunk] = []
    for doc, meta, distance in zip(documents[0], metadatas[0], distances[0]):
        chunks.append(
            RetrievedChunk(
                text=doc,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", ""),
                chunk_id=meta.get("chunk_id", -1),
                page_number=meta.get("page_number", -1),
                score=1 - distance,
            )
        )

    logger.info(
        "Retrieved %d chunks for question (top_k=%d, document_id=%s)",
        len(chunks), top_k, document_id,
    )
    return chunks
