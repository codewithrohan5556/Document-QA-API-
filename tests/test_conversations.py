"""
Tests for GET /conversations/{session_id}.

PostgreSQL is faked via the shared fake_pg_pool fixture (conftest.py),
seeded directly through conversation_service.save_conversation() so these
tests exercise the real INSERT -> SELECT round trip through our own SQL,
not a shortcut that bypasses it.
"""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.database import ConversationRecord
from app.services import conversation_service


def _make_record(session_id: str, question: str, **overrides) -> ConversationRecord:
    defaults = dict(
        session_id=session_id,
        document_id="doc-abc",
        question=question,
        answer=f"Answer to: {question}",
        sources=["file.txt, chunk 0"],
        retrieved_chunk_ids=[0],
        similarity_scores=[0.88],
        model="test-model",
        tokens_used=None,
        latency_ms=250.0,
    )
    defaults.update(overrides)
    return ConversationRecord(**defaults)


async def _get(session_id: str):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(f"/conversations/{session_id}")


@pytest.mark.asyncio
async def test_get_conversation_history_returns_records_for_session(monkeypatch, fake_pg_pool):
    monkeypatch.setattr(conversation_service, "get_pool", lambda: fake_pg_pool)

    now = datetime.now(timezone.utc)
    await conversation_service.save_conversation(
        _make_record("sess-1", "First question?", timestamp=now)
    )
    await conversation_service.save_conversation(
        _make_record("sess-1", "Second question?", timestamp=now + timedelta(seconds=1))
    )
    await conversation_service.save_conversation(
        _make_record("sess-2", "A different session's question", timestamp=now)
    )

    response = await _get("sess-1")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "sess-1"
    assert len(body["conversations"]) == 2
    # oldest first
    assert body["conversations"][0]["question"] == "First question?"
    assert body["conversations"][1]["question"] == "Second question?"
    assert body["conversations"][0]["document_id"] == "doc-abc"
    assert body["conversations"][0]["model"] == "test-model"
    assert "latency_ms" in body["conversations"][0]


@pytest.mark.asyncio
async def test_get_conversation_history_empty_session_returns_empty_list(monkeypatch, fake_pg_pool):
    monkeypatch.setattr(conversation_service, "get_pool", lambda: fake_pg_pool)
    await conversation_service.save_conversation(_make_record("sess-1", "Some question"))

    # sess-999 exists in no rows — legitimately empty, distinct from
    # "Postgres isn't configured" (see next test).
    response = await _get("sess-999")

    assert response.status_code == 200
    assert response.json()["conversations"] == []


@pytest.mark.asyncio
async def test_get_conversation_history_returns_503_when_postgres_not_configured(monkeypatch):
    monkeypatch.setattr(conversation_service, "get_pool", lambda: None)

    response = await _get("sess-1")

    assert response.status_code == 503
    assert "not configured" in response.json()["error"].lower()


@pytest.mark.asyncio
async def test_get_conversation_history_returns_503_on_read_failure(monkeypatch):
    class _BrokenPool:
        def acquire(self):
            raise RuntimeError("postgres connection reset")

    monkeypatch.setattr(conversation_service, "get_pool", lambda: _BrokenPool())

    response = await _get("sess-1")

    assert response.status_code == 503
