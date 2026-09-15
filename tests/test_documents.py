"""
Tests for POST /upload.

Validation, extraction, and chunking are pure local logic with no external
calls. Embedding IS an external call (Groq/OpenRouter over the network), so
we mock `get_embeddings()` here rather than hitting a real provider in
tests — this is the mocking point the project spec calls out. ChromaDB
itself is NOT mocked: it's a local embedded database, so tests exercise the
real add/query/delete code path against a real (test-local) collection.
"""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class _FakeEmbeddings:
    """Deterministic stand-in for OpenAIEmbeddings — no network call."""

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


async def _upload(filename: str, content: bytes, content_type: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/upload",
            files={"file": (filename, content, content_type)},
        )


@pytest.mark.asyncio
async def test_upload_txt_file_returns_indexed_status():
    content = ("This is a test document. " * 100).encode("utf-8")
    response = await _upload("notes.txt", content, "text/plain")

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["status"] == "indexed"
    assert body["chunks_created"] >= 1
    assert len(body["document_id"]) == 12


@pytest.mark.asyncio
async def test_upload_unsupported_file_type_returns_415():
    response = await _upload("image.png", b"fake-bytes", "image/png")

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["error"]


@pytest.mark.asyncio
async def test_upload_empty_file_returns_400():
    response = await _upload("empty.txt", b"", "text/plain")

    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_upload_whitespace_only_txt_returns_400():
    response = await _upload("blank.txt", b"   \n\n   ", "text/plain")

    assert response.status_code == 400
    assert "no extractable text" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_chunking_respects_configured_chunk_size():
    # Long enough to guarantee multiple chunks at the default chunk_size (1000).
    content = ("word " * 2000).encode("utf-8")
    response = await _upload("long.txt", content, "text/plain")

    assert response.status_code == 200
    assert response.json()["chunks_created"] > 1


@pytest.mark.asyncio
async def test_upload_embedding_failure_returns_502(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("provider unreachable")

    broken = MagicMock()
    broken.embed_documents.side_effect = _boom
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embeddings", MagicMock(return_value=broken)
    )

    response = await _upload("notes2.txt", b"Some real content here.", "text/plain")

    assert response.status_code == 502
    assert "embed" in response.json()["error"].lower()


# --- DELETE /document/{document_id} ----------------------------------------

async def _delete(document_id: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.delete(f"/document/{document_id}")


@pytest.mark.asyncio
async def test_delete_document_removes_it_from_vector_store():
    doc_id_response = await _upload("to_delete.txt", ("Delete me. " * 60).encode(), "text/plain")
    doc_id = doc_id_response.json()["document_id"]

    response = await _delete(doc_id)

    assert response.status_code == 200
    body = response.json()
    assert body["document_id"] == doc_id
    assert body["chunks_deleted"] >= 1
    assert body["status"] == "deleted"

    # Prove it's actually gone, not just that the endpoint said 200 —
    # a retrieval scoped to this document_id should now find nothing.
    from app.services.retrieval_service import retrieve_chunks

    results = retrieve_chunks("anything", document_id=doc_id, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_delete_nonexistent_document_returns_404():
    response = await _delete("does-not-exist-123")

    assert response.status_code == 404
    assert "no document found" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_delete_is_idempotent_second_call_returns_404():
    doc_id_response = await _upload("delete_twice.txt", ("Only once. " * 60).encode(), "text/plain")
    doc_id = doc_id_response.json()["document_id"]

    first = await _delete(doc_id)
    second = await _delete(doc_id)

    assert first.status_code == 200
    assert second.status_code == 404


# --- GET /stats -------------------------------------------------------------

async def _get_stats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/stats")
    return response.json()


@pytest.mark.asyncio
async def test_stats_reflects_uploaded_documents_and_postgres_counts(monkeypatch):
    from unittest.mock import AsyncMock

    from app.services import conversation_service

    monkeypatch.setattr(
        conversation_service, "count_queries", AsyncMock(return_value=5)
    )
    monkeypatch.setattr(
        conversation_service, "count_distinct_sessions", AsyncMock(return_value=2)
    )

    before = await _get_stats()
    await _upload("stats_doc.txt", ("Stats content. " * 60).encode(), "text/plain")
    after = await _get_stats()

    # Compare deltas rather than absolute counts — the shared test-local
    # ChromaDB collection may already contain documents from earlier tests
    # in this run, so an exact total would make this test order-dependent.
    assert after["total_documents"] == before["total_documents"] + 1
    assert after["total_queries"] == 5
    assert after["total_conversations"] == 2


@pytest.mark.asyncio
async def test_stats_degrades_to_zero_when_postgres_not_configured(monkeypatch):
    from app.services import conversation_service

    monkeypatch.setattr(conversation_service, "get_pool", lambda: None)

    body = await _get_stats()

    # /stats never errors just because Postgres isn't configured — unlike
    # GET /conversations/{session_id}, this is a best-effort dashboard.
    assert body["total_queries"] == 0
    assert body["total_conversations"] == 0
    assert isinstance(body["total_documents"], int)
