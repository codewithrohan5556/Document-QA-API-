"""
Shared test fixtures.

FakePostgresPool mimics the small subset of asyncpg's API
conversation_service.py actually uses: pool.acquire() as an async context
manager yielding a connection with execute()/fetch()/fetchval(). It's an
in-memory stand-in — no real PostgreSQL server needed to test our own
service logic (query construction, graceful degradation, error handling).
"""

import pytest


class _FakeConnection:
    def __init__(self, store: list[dict]):
        self._store = store

    async def execute(self, query: str, *args):
        if query.strip().upper().startswith("INSERT"):
            (
                session_id, document_id, question, answer, sources,
                retrieved_chunk_ids, similarity_scores, model, tokens_used,
                latency_ms, timestamp,
            ) = args
            self._store.append(
                {
                    "session_id": session_id,
                    "document_id": document_id,
                    "question": question,
                    "answer": answer,
                    "sources": list(sources),
                    "retrieved_chunk_ids": list(retrieved_chunk_ids),
                    "similarity_scores": list(similarity_scores),
                    "model": model,
                    "tokens_used": tokens_used,
                    "latency_ms": latency_ms,
                    "timestamp": timestamp,
                }
            )
        return "OK"

    async def fetch(self, query: str, session_id: str, limit: int):
        matched = [r for r in self._store if r["session_id"] == session_id]
        matched.sort(key=lambda r: r["timestamp"])
        return matched[:limit]

    async def fetchval(self, query: str, *args):
        if "COUNT(DISTINCT session_id)" in query:
            return len({r["session_id"] for r in self._store})
        if "COUNT(*)" in query:
            return len(self._store)
        return 0


class _AcquireContext:
    def __init__(self, conn: _FakeConnection):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc_info):
        return False


class FakePostgresPool:
    """Drop-in stand-in for asyncpg.Pool, backed by an in-memory list."""

    def __init__(self):
        self.store: list[dict] = []

    def acquire(self):
        return _AcquireContext(_FakeConnection(self.store))


@pytest.fixture
def fake_pg_pool():
    return FakePostgresPool()
