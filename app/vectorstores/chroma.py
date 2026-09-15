"""
ChromaDB implementation of VectorStore.

ChromaDB is a local, embedded vector database — no server to run, persists
to disk at settings.chroma_persist_dir. This is our Phase 1 backend; nothing
outside this file and vectorstores/__init__.py knows Chroma exists.
"""

from typing import Any

import chromadb

from app.core.config import settings
from app.core.logging import get_logger
from app.vectorstores.base import VectorStore

logger = get_logger(__name__)

COLLECTION_NAME = "documents"


class ChromaVectorStore(VectorStore):
    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        # Explicit cosine distance: our embedding vectors (Groq/NVIDIA-compatible
        # OpenAI embeddings) are meant to be compared via cosine similarity, not
        # Chroma's default squared-L2. Without this, "distance" wouldn't map
        # cleanly to a similarity score in retrieval_service.py.
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready (path=%s, collection=%s, count=%d)",
            settings.chroma_persist_dir, COLLECTION_NAME, self._collection.count(),
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self._collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Cap top_k to the number of chunks actually available in the
        # filtered collection — Chroma's HNSW index raises a RuntimeError
        # if asked for more results than exist ("Cannot return the results
        # in a contiguous 2D array"). This is common in tests (small
        # documents, top_k=5) and can also happen in real use when a
        # document has fewer chunks than the configured top_k.
        if filters:
            available = len(self._collection.get(where=filters)["ids"])
        else:
            available = self._collection.count()

        n_results = min(top_k, available)
        if n_results == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=filters,
        )

    def delete(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id})
        deleted_count = len(existing["ids"])
        if deleted_count:
            self._collection.delete(where={"document_id": document_id})
        return deleted_count

    def count_documents(self) -> int:
        all_metadata = self._collection.get(include=["metadatas"])["metadatas"]
        distinct_ids = {meta["document_id"] for meta in all_metadata}
        return len(distinct_ids)
