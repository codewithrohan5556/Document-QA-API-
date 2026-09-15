"""
Tests for page-aware source citations (Step 10).

Uses reportlab to generate a REAL multi-page PDF with actual extractable
text. The blank PDFs used in earlier manual checks (Step 6) can't prove
page-number citation logic — there's no text on those pages to extract, so
every page would just get filtered out by extract_pages().
"""

import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from reportlab.pdfgen import canvas

from app.main import app
from app.services import document_service, ingestion_service


def _make_multipage_pdf(page_texts: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in page_texts:
        c.drawString(72, 700, text)
        c.showPage()
    c.save()
    return buf.getvalue()


class _FakeEmbeddings:
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


# --- Unit-level: document_service.extract_pages ---------------------------

def test_extract_pages_returns_one_entry_per_pdf_page():
    pdf_bytes = _make_multipage_pdf(
        ["First page content.", "Second page content.", "Third page content."]
    )
    pages = document_service.extract_pages(pdf_bytes, ".pdf")

    assert len(pages) == 3
    assert "First page" in pages[0]
    assert "Second page" in pages[1]
    assert "Third page" in pages[2]


def test_extract_pages_txt_has_a_single_page():
    pages = document_service.extract_pages(b"Just some plain text content.", ".txt")
    assert len(pages) == 1


# --- Unit-level: ingestion_service.chunk_pages -----------------------------

def test_chunk_pages_tags_pdf_chunks_with_real_page_numbers():
    pages = ["Page one text. " * 5, "Page two text. " * 5]
    chunks = ingestion_service.chunk_pages(pages, ".pdf", chunk_size=1000, chunk_overlap=100)

    page_numbers = {page_number for page_number, _ in chunks}
    assert page_numbers == {1, 2}


def test_chunk_pages_txt_uses_no_page_sentinel():
    pages = ["Just plain text, no real pages here."]
    chunks = ingestion_service.chunk_pages(pages, ".txt")

    assert all(page_number == -1 for page_number, _ in chunks)


# --- End-to-end: upload a real PDF, query it, check citations -------------

async def _upload_pdf(filename: str, pdf_bytes: bytes) -> str:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/upload", files={"file": (filename, pdf_bytes, "application/pdf")}
        )
    assert response.status_code == 200
    return response.json()["document_id"]


@pytest.mark.asyncio
async def test_upload_pdf_and_query_cites_real_page_numbers(monkeypatch):
    pdf_bytes = _make_multipage_pdf(
        [
            "Photosynthesis converts light into chemical energy. " * 10,
            "Chlorophyll absorbs light inside plant leaves. " * 10,
        ]
    )
    doc_id = await _upload_pdf("bio.pdf", pdf_bytes)

    class _FakeChunk:
        def __init__(self, content: str):
            self.content = content

    async def fake_astream(prompt):
        yield _FakeChunk("Plants use chlorophyll to capture light. ")

    fake_llm = MagicMock()
    fake_llm.astream = fake_astream
    monkeypatch.setattr("app.services.rag_service.get_llm", MagicMock(return_value=fake_llm))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST", "/query",
            json={"document_id": doc_id, "question": "How do plants get energy?"},
        ) as response:
            body = ""
            async for text in response.aiter_text():
                body += text

    assert response.status_code == 200
    assert "Sources:" in body
    sources_block = body.split("Sources:")[-1]
    assert "bio.pdf, page" in sources_block
    # Real PDF pages should never fall back to chunk-based citations.
    assert "chunk" not in sources_block


@pytest.mark.asyncio
async def test_generate_answer_txt_still_falls_back_to_chunk_citations(monkeypatch):
    from app.services.rag_service import generate_answer

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        upload_resp = await client.post(
            "/upload",
            files={"file": ("plain.txt", ("Just plain text content. " * 60).encode(), "text/plain")},
        )
    doc_id = upload_resp.json()["document_id"]

    class _FakeResponse:
        content = "This is plain text. [chunk 0]"

    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=_FakeResponse())
    monkeypatch.setattr("app.services.rag_service.get_llm", MagicMock(return_value=fake_llm))

    result = await generate_answer("What is this?", document_id=doc_id, top_k=3)

    assert result.sources[0].startswith("plain.txt, chunk")
    assert "page" not in result.sources[0]
