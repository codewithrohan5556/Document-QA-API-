"""
Vector store factory.

The ONE place in the codebase that knows about VECTOR_STORE=chroma vs
VECTOR_STORE=pinecone. Every other module calls `get_vector_store()` and
gets back something that satisfies the `VectorStore` interface — it never
imports ChromaVectorStore or PineconeVectorStore itself.

Phase 1: VECTOR_STORE=chroma  -> ChromaVectorStore (local embedded DB)
Phase 2: VECTOR_STORE=pinecone -> PineconeVectorStore (managed cloud DB)

Switching from Chroma to Pinecone requires:
  1. Set VECTOR_STORE=pinecone in .env
  2. Set PINECONE_API_KEY in .env
  3. Re-upload any documents (the two backends don't share a data store)

Nothing in services/ or routers/ changes.
"""

from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger
from app.vectorstores.base import VectorStore

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    if settings.vector_store == "chroma":
        from app.vectorstores.chroma import ChromaVectorStore

        return ChromaVectorStore()

    if settings.vector_store == "pinecone":
        from app.vectorstores.pinecone import PineconeVectorStore

        return PineconeVectorStore()

    raise ValueError(
        f"Unknown VECTOR_STORE '{settings.vector_store}'. Expected 'chroma' or 'pinecone'."
    )
