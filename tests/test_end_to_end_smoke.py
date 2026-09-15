"""
End-to-end smoke test (Step 18 — ChromaDB Validation Checklist).

Walks the full pipeline in one flowing test: upload -> chunk -> embed ->
store -> query -> retrieve -> generate -> stream -> cite -> persist ->
read history -> stats -> delete -> confirm gone -> error handling.

Only three things are mocked, same as every other test in this suite:
Groq (LLM), OpenRouter (embeddings), and PostgreSQL (via the shared
fake_pg_pool fixture). Everything else — FastAPI routing, real ChromaDB
storage/retrieval, page-aware chunking and citations, streaming mechanics —
runs for real. This is deliberately NOT a shortcut version of the other
test files; it exists to prove the pieces work TOGETHER, in the order a
real user would actually hit them, not just individually in isolation.
"""

import io
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from app.main import app
from app.services import conversation_service


def _make_pdf(page_texts: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in page_texts:
        c.drawString(72, 700, text)
        c.showPage()
    c.save()
    return buf.getvalue()


class _FakeEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class _FakeChunk:
    def __init__(self, content: str):
        self.content = content


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(monkeypatch, fake_pg_pool):
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.services.ingestion_service.get_embeddings", MagicMock(return_value=fake_embeddings)
    )
    monkeypatch.setattr(
        "app.services.retrieval_service.get_embeddings", MagicMock(return_value=fake_embeddings)
    )
    monkeypatch.setattr(conversation_service, "get_pool", lambda: fake_pg_pool)

    transport = ASGITransport(app=app)

    # --- Checklist: "Documents can be uploaded" / "correctly chunked" ---
    pdf_bytes = _make_pdf(
        [
            "The Great Barrier Reef is the world's largest coral reef system. " * 6,
            "It is located off the coast of Queensland, Australia. " * 6,
        ]
    )
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        upload_resp = await client.post(
            "/upload", files={"file": ("reef.pdf", pdf_bytes, "application/pdf")}
        )
    assert upload_resp.status_code == 200
    upload_body = upload_resp.json()
    assert upload_body["status"] == "indexed"
    assert upload_body["chunks_created"] == 2  # one chunk per page at this length
    doc_id = upload_body["document_id"]

    # --- Checklist: "Embeddings are generated" / "Chunks are stored in
    # ChromaDB" / "Metadata is preserved" — verified by reaching directly
    # into the real Chroma collection, not just trusting the API response.
    # This is deliberately validation-only; normal code never does this.
    from app.vectorstores import get_vector_store

    store = get_vector_store()
    raw = store._collection.get(where={"document_id": doc_id}, include=["metadatas"])
    assert len(raw["metadatas"]) == 2
    for meta in raw["metadatas"]:
        assert meta["document_id"] == doc_id
        assert meta["filename"] == "reef.pdf"
        assert meta["source"] == "reef.pdf"
        assert meta["page_number"] in (1, 2)

    # --- Checklist: "Queries retrieve relevant chunks" / "Answers are
    # grounded in retrieved context" / "Source citations work" / "Streaming
    # works" ---
    async def fake_astream(prompt):
        # Proves grounding: the actual page content made it into the prompt
        # sent to the LLM, not just some placeholder.
        assert "Great Barrier Reef" in prompt
        assert "Queensland" in prompt
        for token in ["The Great Barrier Reef ", "is off the coast ", "of Queensland."]:
            yield _FakeChunk(token)

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr("app.services.rag_service.get_llm", MagicMock(return_value=fake_llm))

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/query",
            json={"document_id": doc_id, "question": "Where is the reef?", "session_id": "smoke-1"},
        ) as response:
            assert response.status_code == 200
            body = ""
            async for text in response.aiter_text():
                body += text

    assert "Great Barrier Reef" in body
    assert "Queensland" in body
    assert "Sources:" in body
    assert "reef.pdf, page" in body
    # Note: this doesn't assert on the number of pieces httpx's in-process
    # ASGITransport delivers them in — that's a property of the test
    # transport (which can coalesce fast successive writes with no real
    # network latency between them), not evidence about our streaming
    # implementation. The streaming mechanism itself is already proven at
    # the architecture level by test_query.py::test_query_streams_tokens_and_sources.

    # --- Checklist: "PostgreSQL stores conversations" (replaces "MongoDB
    # stores conversations" post-migration) ---
    assert len(fake_pg_pool.store) == 1
    saved = fake_pg_pool.store[0]
    assert saved["session_id"] == "smoke-1"
    assert saved["document_id"] == doc_id
    assert "Sources:" not in saved["answer"]  # trailing block stripped before storage

    # --- Checklist: "Conversation history works" ---
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        history_resp = await client.get("/conversations/smoke-1")
    assert history_resp.status_code == 200
    conversations = history_resp.json()["conversations"]
    assert len(conversations) == 1
    assert conversations[0]["question"] == "Where is the reef?"

    # --- Checklist: "Stats endpoint works" ---
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        stats_resp = await client.get("/stats")
    stats_body = stats_resp.json()
    assert stats_body["total_queries"] == 1
    assert stats_body["total_conversations"] == 1
    assert stats_body["total_documents"] >= 1

    # --- Checklist: "Error handling works" ---
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad_resp = await client.post(
            "/upload", files={"file": ("bad.exe", b"not a real doc", "application/octet-stream")}
        )
    assert bad_resp.status_code == 415
    assert "error" in bad_resp.json()

    # --- Checklist: "Document deletion works" — and prove it's REALLY
    # gone, not just a 200 response ---
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        delete_resp = await client.delete(f"/document/{doc_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["chunks_deleted"] == 2

    from app.services.retrieval_service import retrieve_chunks

    remaining = retrieve_chunks("anything", document_id=doc_id, top_k=5)
    assert remaining == []

    # Deleting again must 404, not silently succeed a second time.
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        second_delete_resp = await client.delete(f"/document/{doc_id}")
    assert second_delete_resp.status_code == 404
