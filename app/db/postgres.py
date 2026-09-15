"""
PostgreSQL connection management via asyncpg.

One connection pool per process, opened at app startup and closed at
shutdown via the FastAPI lifespan hook in main.py — same pattern already
used for configure_logging(). Nothing outside this module touches the raw
pool; services call get_pool() instead.

"""

import asyncpg

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None

# Native Postgres arrays (TEXT[], INTEGER[], DOUBLE PRECISION[]) for the
# list fields — asyncpg converts these to/from plain Python lists
# automatically, no JSON encode/decode needed, and the column types are
# self-documenting in the schema itself.
CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT[] NOT NULL DEFAULT '{}',
    retrieved_chunk_ids INTEGER[] NOT NULL DEFAULT '{}',
    similarity_scores DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
    model TEXT NOT NULL,
    tokens_used INTEGER,
    latency_ms DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations (session_id);
"""


async def connect_to_postgres() -> None:
    global _pool

    if not settings.postgres_dsn:
        logger.warning("POSTGRES_DSN is not set — conversation history will not be persisted.")
        return

    try:
        _pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=10)
        async with _pool.acquire() as conn:
            await conn.execute(CREATE_CONVERSATIONS_TABLE)
        logger.info("Connected to PostgreSQL and ensured schema exists")
    except Exception as exc:
        logger.error("Failed to connect to PostgreSQL: %s", exc)
        _pool = None


async def close_postgres_connection() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Closed PostgreSQL connection pool")


def get_pool() -> asyncpg.Pool | None:
    """
    Returns the active connection pool, or None if Postgres isn't
    configured or failed to connect. Callers (conversation_service.py) must
    handle the None case gracefully.
    """
    return _pool
