"""
The VectorStore abstraction.

This is the mandatory piece from the project spec: retrieval_service and
rag_service  will import `VectorStore` and `get_vector_store()`.
"""
from abc import ABC,abstractmethod
from typing import Any

class VectorStore(ABC):
    @abstractmethod
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Store chunks with their embeddings and metadata."""
        ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Similarity search. Returns a dict with (at minimum) "documents",
        "metadatas", and "distances" — matching Chroma's query() shape,
        since that's the lowest common denominator both backends can produce.
        """
        ...

    @abstractmethod
    def delete(self, document_id: str) -> int:
        """
        Remove every chunk belonging to a given document_id.

        Returns the number of chunks actually deleted (0 if the document_id
        didn't exist). This return value is what DELETE /document/{id} uses
        as its existence check — no separate "does this document exist"
        lookup needed, and no dedicated document-metadata store required
        just to answer that question.
        """
        ...

    @abstractmethod
    def count_documents(self) -> int:
        """
        Number of distinct document_ids currently stored — powers GET /stats.

        Note: this is a reasonable approach at the scale of a course/portfolio
        project (pull all metadata, count distinct document_ids in memory),
        not something that would scale to a large production corpus, where
        you'd want the backend's native aggregation instead. Documented
        limitation, not a hidden one.
        """
        ...
