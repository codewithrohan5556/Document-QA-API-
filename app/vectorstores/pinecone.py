"""
Pinecone implementation of VectorStore (Phase 2).

Pinecone is a managed, cloud-hosted vector database — the production-grade
alternative to ChromaDB's local embedded storage. This implementation
satisfies the same VectorStore interface (base.py) that ChromaVectorStore
does, so nothing in retrieval_service.py, ingestion_service.py, or
rag_service.py changes when switching backends. The only change is setting
VECTOR_STORE=pinecone in .env and providing a PINECONE_API_KEY — the
factory in vectorstores/__init__.py does the rest.

Key differences from ChromaDB worth understanding:
  - Pinecone is a remote API, not an embedded library. Every add/query/delete
    is an HTTP request, so there is real network latency. For a portfolio
    project this is fine; at production scale you'd batch upserts.
  - Pinecone stores metadata alongside vectors (same as Chroma) but has a
    simpler filter syntax: {"key": {"$eq": "value"}} instead of Chroma's
    {"key": "value"} shorthand. The base.py interface's `filters` dict is
    the Chroma shorthand shape, so we translate it here.
  - Pinecone's index must be pre-created (via the console or SDK) before
    first use — dimension and metric are set at creation time, not per
    request. We create it automatically if it doesn't exist.
  - Pinecone doesn't directly store the original chunk text alongside the
    vector (only metadata). We store the text in a "text" metadata field
    as a workaround — same pattern many production systems use.
  - delete() counts deleted vectors by querying before deletion, because
    Pinecone's delete() response doesn't return a count (unlike Chroma's).
"""

from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings
from app.core.logging import get_logger
from app.vectorstores.base import VectorStore

logger = get_logger(__name__)


class PineconeVectorStore(VectorStore):
    def __init__(self):
        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is not set. "
                "Set it in .env before using VECTOR_STORE=pinecone."
            )

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._dimension = settings.pinecone_dimension

        self._ensure_index_exists()
        self._index = self._pc.Index(self._index_name)

        stats = self._index.describe_index_stats()
        logger.info(
            "PineconeVectorStore ready (index=%s, dimension=%d, total_vectors=%d)",
            self._index_name, self._dimension, stats.total_vector_count,
        )

    def _ensure_index_exists(self) -> None:
        """
        Creates the index if it doesn't already exist. The dimension and
        metric are set here at index creation time — they can't be changed
        afterward without deleting and recreating the index.

        Uses the serverless spec (free tier compatible) on AWS us-east-1.
        Adjust cloud/region to match your Pinecone account's available
        regions if needed.
        """
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing:
            logger.info(
                "Pinecone index '%s' not found — creating (dimension=%d, metric=cosine)",
                self._index_name, self._dimension,
            )
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Pinecone index '%s' created successfully.", self._index_name)


    def _to_pinecone_filter(
        self, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Translates Chroma-style flat filters {"document_id": "abc"} into
        Pinecone's filter syntax {"document_id": {"$eq": "abc"}}.

        The base.py interface uses Chroma's shorthand since that was the
        Phase 1 implementation. This translation keeps the interface
        consistent while meeting Pinecone's actual filter requirements.
        """
        if not filters:
            return None
        return {key: {"$eq": value} for key, value in filters.items()}


    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Upserts vectors in batches of 100.

        Pinecone's upsert API accepts up to 100 vectors per request in the
        free tier. We store the chunk text in metadata["text"] since
        Pinecone doesn't have a separate document storage field.
        """
        vectors = []
        for chunk_id, (embedding, text, metadata) in enumerate(
            zip(embeddings, documents, metadatas)
        ):
            vectors.append({
                "id": ids[chunk_id],
                "values": embedding,
                "metadata": {**metadata, "text": text},
            })

        # Batch upsert in groups of 100 (free-tier limit)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch)

        logger.info("Upserted %d vectors to Pinecone index '%s'", len(vectors), self._index_name)


    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Queries for the top-k most similar vectors.

        Returns a dict in the same shape as ChromaDB's query() output —
        {"documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        — so retrieval_service.py can consume it without knowing which
        backend produced it. This shape normalization is the key contract
        both implementations must honor.

        Pinecone returns similarity scores (higher = more similar), while
        Chroma returns distances (lower = more similar). We convert here:
        distance = 1 - score, matching how retrieval_service.py later
        converts back to score = 1 - distance. The round-trip is a no-op
        numerically but keeps the interface consistent.
        """
        pinecone_filter = self._to_pinecone_filter(filters)

        response = self._index.query(
            vector=embedding,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True,
        )

        documents = []
        metadatas = []
        distances = []

        for match in response.matches:
            meta = dict(match.metadata)
            text = meta.pop("text", "")
            documents.append(text)
            metadatas.append(meta)
            # Convert Pinecone similarity score -> distance for consistency
            distances.append(1 - match.score)

        return {
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }


    def delete(self, document_id: str) -> int:
        """
        Deletes all vectors belonging to document_id.

        Pinecone's delete() doesn't return a count, so we query first
        to count matching vectors, then delete by ID. This is two API
        calls instead of Chroma's one, but it's the only reliable way
        to get a real deleted count for the DELETE /document/{id}
        endpoint's response.
        """
        pinecone_filter = self._to_pinecone_filter({"document_id": document_id})

        # Use a dummy vector to find all matching IDs via metadata filter.
        # We use a zero vector because we only care about the metadata match,
        # not the actual similarity ordering.
        dummy_vector = [0.0] * self._dimension
        response = self._index.query(
            vector=dummy_vector,
            top_k=10000,
            filter=pinecone_filter,
            include_metadata=False,
        )

        ids_to_delete = [match.id for match in response.matches]
        if not ids_to_delete:
            return 0

        self._index.delete(ids=ids_to_delete)
        logger.info(
            "Deleted %d vectors for document_id=%s from Pinecone index '%s'",
            len(ids_to_delete), document_id, self._index_name,
        )
        return len(ids_to_delete)


    def count_documents(self) -> int:
        """
        Returns the number of distinct document_ids currently in the index.

        Pinecone doesn't support GROUP BY or aggregate queries, so we use
        describe_index_stats() to get namespace stats, then fall back to a
        metadata scan if needed. For the scale of this project (tens of
        documents) this is fine. For a large corpus you'd maintain a
        separate document registry.
        """
        stats = self._index.describe_index_stats()
        total = stats.total_vector_count
        if total == 0:
            return 0

        # Scan all vectors' metadata to count distinct document_ids.
        # We use a zero vector and large top_k as a proxy for "fetch all".
        dummy_vector = [0.0] * self._dimension
        response = self._index.query(
            vector=dummy_vector,
            top_k=min(total, 10000),
            include_metadata=True,
        )
        distinct_ids = {
            match.metadata.get("document_id")
            for match in response.matches
            if match.metadata.get("document_id")
        }
        return len(distinct_ids)
