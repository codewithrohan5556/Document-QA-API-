"""
Tests for POST /query.

Covers: normal streaming, the no-matching-chunks case, a failure mid-stream
(after tokens have already started), and — the important one — a retrieval
failure that happens BEFORE streaming starts, proving it still returns a
clean JSON error rather than a broken stream. That last test is what
justifies retrieval happening in the router rather than inside
stream_answer_tokens() (see that function's docstring).
"""

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeStreamChunk:
    def __init__(self, content: str):
        self.content = content


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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload", files={"file": (filename, content, "text/plain")}
        )
    assert response.status_code == 200
    return response.json()["document_id"]


async def _stream_query(payload: dict) -> tuple[int, str]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/query", json=payload) as response:
            body = ""
            async for text in response.aiter_text():
                body += text
            return response.status_code, body


@pytest.mark.asyncio
async def test_query_streams_tokens_and_sources(monkeypatch):
    doc_id = await _upload("stream_a.txt", ("Streaming test content. " * 60).encode())

    async def fake_astream(prompt):
        for token in ["The ", "document ", "is ", "about ", "testing."]:
            yield _FakeStreamChunk(token)

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=fake_llm)
    )

    status, body = await _stream_query(
        {"document_id": doc_id, "question": "What is this about?", "session_id": "s1"}
    )

    assert status == 200
    assert "The document is about testing." in body
    assert "Sources:" in body
    assert "stream_a.txt, chunk" in body


@pytest.mark.asyncio
async def test_query_with_no_matching_document_returns_no_context_message(monkeypatch):
    calls = []

    async def fake_astream(prompt):
        calls.append(prompt)
        yield _FakeStreamChunk("should never run")

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=fake_llm)
    )

    status, body = await _stream_query(
        {"document_id": "never-uploaded-doc", "question": "Anything?", "session_id": "s1"}
    )

    assert status == 200
    assert "I don't have enough information" in body
    assert calls == []  # LLM never called when there's no context


@pytest.mark.asyncio
async def test_query_mid_stream_failure_yields_inline_error(monkeypatch):
    doc_id = await _upload("stream_b.txt", ("More streaming content. " * 60).encode())

    async def fake_astream(prompt):
        yield _FakeStreamChunk("Partial answer... ")
        raise RuntimeError("groq connection dropped")

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=fake_llm)
    )

    status, body = await _stream_query(
        {"document_id": doc_id, "question": "What is this?", "session_id": "s1"}
    )

    # Headers were already sent with 200 before the failure occurred mid-stream —
    # this is expected and is exactly the scenario the docstring in
    # rag_service.stream_answer() describes.
    assert status == 200
    assert "Partial answer..." in body
    assert "[Error:" in body


@pytest.mark.asyncio
async def test_query_retrieval_failure_returns_clean_json_error(monkeypatch):
    """
    Proves the design decision: a failure during retrieval (before the
    stream starts) still becomes a proper JSON error response, because
    retrieve_chunks() runs in the router, outside the generator.
    """
    from app.utils.exceptions import VectorStoreError

    def broken_retrieve(*args, **kwargs):
        raise VectorStoreError("vector database is unreachable")

    monkeypatch.setattr("app.routers.query.retrieve_chunks", broken_retrieve)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"document_id": "doesnt-matter", "question": "Anything?", "session_id": "s1"},
        )

    assert response.status_code == 502
    assert "vector database is unreachable" in response.json()["error"]


@pytest.mark.asyncio
async def test_query_rejects_empty_question():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"document_id": "doc1", "question": "", "session_id": "s1"},
        )

    assert response.status_code == 422  # Pydantic validation, clean JSON, no traceback


@pytest.mark.asyncio
async def test_query_missing_document_id_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"question": "What is this?"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_missing_question_field_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"document_id": "doc1"})

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_persists_conversation_record_after_streaming(monkeypatch):
    """
    Step 11: proves /query actually calls through to conversation persistence
    with the right data, once streaming has finished. PostgreSQL itself is
    mocked here too (conversation_service already has its own unit tests
    against a real-shaped fake in test_conversation_service.py) — this test
    is about the wiring between query.py -> rag_service -> conversation_service,
    not about Postgres.
    """
    doc_id = await _upload("stream_c.txt", ("Persist this content. " * 60).encode())

    async def fake_astream(prompt):
        yield _FakeStreamChunk("The answer is 42.")

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=fake_llm)
    )

    saved = {}

    async def fake_save(record):
        saved["record"] = record

    monkeypatch.setattr(
        "app.services.conversation_service.save_conversation", fake_save
    )

    status, body = await _stream_query(
        {"document_id": doc_id, "question": "What is the answer?", "session_id": "sess-42"}
    )

    assert status == 200
    assert "record" in saved  # save_conversation was actually called
    record = saved["record"]
    assert record.session_id == "sess-42"
    assert record.document_id == doc_id
    assert record.question == "What is the answer?"
    assert "The answer is 42." in record.answer
    assert "Sources:" not in record.answer  # trailing sources block stripped back off
    assert record.latency_ms >= 0
    assert len(record.retrieved_chunk_ids) > 0
    assert len(record.similarity_scores) == len(record.retrieved_chunk_ids)
