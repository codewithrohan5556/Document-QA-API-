"""
Text chunking, embedding, and indexing. This is intentionally the same
hand-rolled overlap-chunking approach from the understanding notebook — no
LangChain text splitter yet. We'll swap in RecursiveCharacterTextSplitter
once LangChain is doing more for us (Step 7+); for now this keeps the logic
visible and testable in isolation, and `chunk_size`/`chunk_overlap` are never
hardcoded — they come from settings (or explicit overrides) so behavior is
configurable from one place, per the project's retrieval-pipeline requirement.
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.services.embedding_service import get_embeddings
from app.utils.exceptions import VectorStoreError
from app.vectorstores import get_vector_store

logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size (or chunking never advances)")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    logger.info(
        "Chunked %d chars into %d chunks (chunk_size=%d, overlap=%d)",
        len(text), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def chunk_pages(
    pages: list[str],
    ext: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[tuple[int, str]]:
    """
    Chunks each page independently and tags every resulting chunk with a
    page number, returned as (page_number, chunk_text) pairs.

    A chunk never straddles a page boundary — a deliberate trade-off: it can
    slightly increase chunk count right at page edges, but keeps citations
    honestly page-accurate instead of guessing which page a merged chunk
    "mostly" came from.

    page_number is only meaningful for PDFs, which have a real page concept.
    For .txt (single "page" = the whole file), we use -1 as a sentinel for
    "no real page here" — Chroma metadata can't store None, and rag_service
    uses this sentinel to fall back to chunk-based citations instead of
    claiming a fake "page 1".
    """
    result: list[tuple[int, str]] = []
    for index, page_text in enumerate(pages, start=1):
        page_number = index if ext == ".pdf" else -1
        for chunk in chunk_text(page_text, chunk_size, chunk_overlap):
            result.append((page_number, chunk))
    return result


def embed_and_index_chunks(
    document_id: str, filename: str, chunk_pages: list[tuple[int, str]]
) -> None:
    """
    Embeds every chunk and stores it in the active vector store.

    Deliberately depends on `get_vector_store()` (the VectorStore interface)
    and `get_embeddings()` (the embedding-provider interface) — never on
    ChromaDB or a specific embedding provider directly. That's what lets
    Phase 2 swap Pinecone in here with zero changes to this function.

    Metadata stored per chunk (document_id, filename, chunk_id, source,
    page_number) is what powers source citations in the query pipeline.
    `chunk_pages` is a list of (page_number, chunk_text) — see chunk_pages()
    above for where page_number comes from.
    """
    texts = [text for _, text in chunk_pages]

    try:
        embeddings_model = get_embeddings()
        vectors = embeddings_model.embed_documents(texts)
    except Exception as exc:
        logger.error("Embedding failed for document_id=%s: %s", document_id, exc)
        raise VectorStoreError(f"Failed to generate embeddings: {exc}") from exc

    ids = [f"{document_id}_{i}" for i in range(len(texts))]
    metadatas = [
        {
            "document_id": document_id,
            "filename": filename,
            "chunk_id": i,
            "source": filename,
            "page_number": page_number,
        }
        for i, (page_number, _) in enumerate(chunk_pages)
    ]

    try:
        vector_store = get_vector_store()
        vector_store.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
    except Exception as exc:
        logger.error("Vector store write failed for document_id=%s: %s", document_id, exc)
        raise VectorStoreError(f"Failed to store document in vector database: {exc}") from exc

    logger.info("Indexed %d chunks for document_id=%s into vector store", len(texts), document_id)
