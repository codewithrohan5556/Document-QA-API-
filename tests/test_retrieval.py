"""
Tests for retrieval_service.retrieve_chunks().

Strategy: upload a real document through the actual /upload endpoint (so we
get a real ChromaDB collection populated with real metadata), then call
retrieve_chunks() directly as a service — not through HTTP, since there's no
/query router yet (that's Step 8+). Embeddings are mocked on both the
ingestion side and the retrieval side, same reasoning as test_documents.py:
no network calls to Groq/OpenRouter in tests.
"""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.retrieval_service import retrieve_chunks


class _FakeEmbeddings:
    """
    Constant vector for every input. Good enough to prove the retrieval
    *pipeline* works end to end (metadata plumbing, filtering, top_k) — not
    meant to prove semantic relevance, which requires a real embedding model.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def mock_embeddings(monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embeddings", MagicMock(return_value=fake)
    )
    monkeypatch.setattr(
        "app.services.retrieval_service.get_embeddings", MagicMock(return_value=fake)
    )
    yield fake


async def _upload(filename: str, content: bytes) -> str:
    """Uploads a document via the real endpoint and returns its document_id."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload", files={"file": (filename, content, "text/plain")}
        )
    assert response.status_code == 200
    return response.json()["document_id"]


@pytest.mark.asyncio
async def test_retrieve_chunks_returns_chunks_for_own_document():
    doc_id = await _upload("retrieval_a.txt", ("Alpha content here. " * 60).encode())

    results = retrieve_chunks("What is this about?", document_id=doc_id, top_k=3)

    assert 1 <= len(results) <= 3
    for chunk in results:
        assert chunk.document_id == doc_id
        assert chunk.filename == "retrieval_a.txt"
        assert chunk.chunk_id >= 0
        assert isinstance(chunk.score, float)


@pytest.mark.asyncio
async def test_retrieve_chunks_respects_top_k():
    doc_id = await _upload("retrieval_b.txt", ("Beta content here. " * 300).encode())

    results = retrieve_chunks("question", document_id=doc_id, top_k=2)

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_retrieve_chunks_filters_by_document_id():
    doc_a = await _upload("retrieval_c.txt", ("Doc A unique content. " * 60).encode())
    doc_b = await _upload("retrieval_d.txt", ("Doc B unique content. " * 60).encode())

    results = retrieve_chunks("question", document_id=doc_a, top_k=10)

    assert len(results) > 0
    assert all(chunk.document_id == doc_a for chunk in results)
    assert all(chunk.document_id != doc_b for chunk in results)


@pytest.mark.asyncio
async def test_retrieve_chunks_without_document_id_searches_everything():
    await _upload("retrieval_e.txt", ("Some content. " * 60).encode())

    results = retrieve_chunks("question", document_id=None, top_k=5)

    assert len(results) > 0
