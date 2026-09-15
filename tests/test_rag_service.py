"""
Tests for rag_service.generate_answer().

Two external calls happen in this pipeline: embeddings (Groq/OpenRouter via
retrieval_service) and the LLM (Groq via llm_service). Both are mocked — no
network calls in tests. ChromaDB itself is real, same as test_retrieval.py:
we upload a real document first so retrieval has real data to find.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services.rag_service import NO_CONTEXT_ANSWER, generate_answer


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _FakeLLMResponse:
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


@pytest.fixture
def mock_llm(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(
        return_value=_FakeLLMResponse("The document is about testing. [chunk 0]")
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=fake_llm)
    )
    return fake_llm


async def _upload(filename: str, content: bytes) -> str:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload", files={"file": (filename, content, "text/plain")}
        )
    assert response.status_code == 200
    return response.json()["document_id"]


@pytest.mark.asyncio
async def test_generate_answer_returns_answer_and_sources(mock_llm):
    doc_id = await _upload("rag_a.txt", ("Testing content for RAG. " * 60).encode())

    result = await generate_answer("What is this about?", document_id=doc_id, top_k=3)

    assert result.answer == "The document is about testing. [chunk 0]"
    assert result.chunks_used >= 1
    assert len(result.sources) >= 1
    assert all(doc_id not in "" for _ in result.sources)  # sanity: sources is non-empty list
    assert result.sources[0].startswith("rag_a.txt, chunk")
    mock_llm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_answer_with_no_matching_chunks_skips_llm_call(mock_llm):
    # A document_id that was never indexed -> retrieve_chunks() finds nothing.
    result = await generate_answer("Anything?", document_id="doesnotexist123", top_k=3)

    assert result.answer == NO_CONTEXT_ANSWER
    assert result.sources == []
    assert result.chunks_used == 0
    mock_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_answer_raises_llm_error_on_failure(monkeypatch):
    doc_id = await _upload("rag_b.txt", ("More testing content. " * 60).encode())

    broken_llm = MagicMock()
    broken_llm.ainvoke = AsyncMock(side_effect=RuntimeError("groq unreachable"))
    monkeypatch.setattr(
        "app.services.rag_service.get_llm", MagicMock(return_value=broken_llm)
    )

    from app.utils.exceptions import LLMError

    with pytest.raises(LLMError):
        await generate_answer("What is this?", document_id=doc_id, top_k=3)
