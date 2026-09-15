"""
Conversation persistence.

Writes one record per completed /query call (Step 11), and reads them back
for GET /conversations/{session_id} (Step 12). Migrated from MongoDB to
PostgreSQL — the graceful-degradation philosophy is unchanged, only the
storage mechanics (SQL instead of document queries) are different.

Note the asymmetry between save_conversation() and get_conversation_history()
— see the latter's docstring for why saving degrades silently but reading
raises loudly.
"""

from app.core.logging import get_logger
from app.db.postgres import get_pool
from app.models.database import ConversationRecord
from app.utils.exceptions import ServiceUnavailableError


logger = get_logger(__name__)

_INSERT_SQL = """
INSERT INTO conversations
    (session_id, document_id, question, answer, sources, retrieved_chunk_ids,
     similarity_scores, model, tokens_used, latency_ms, timestamp)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_SELECT_HISTORY_SQL = """
SELECT session_id, document_id, question, answer, sources, retrieved_chunk_ids,
       similarity_scores, model, tokens_used, latency_ms, timestamp
FROM conversations
WHERE session_id = $1
ORDER BY timestamp ASC
LIMIT $2
"""


async def save_conversation(record: ConversationRecord) -> None:
    pool = get_pool()
    if pool is None:
        logger.debug("PostgreSQL not configured — skipping conversation save.")
        return

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                _INSERT_SQL,
                record.session_id,
                record.document_id,
                record.question,
                record.answer,
                record.sources,
                record.retrieved_chunk_ids,
                record.similarity_scores,
                record.model,
                record.tokens_used,
                record.latency_ms,
                record.timestamp,
            )
    except Exception as exc:
        # Deliberately not re-raised — see module docstring.
        logger.error("Failed to save conversation record: %s", exc)


async def get_conversation_history(session_id: str, limit: int = 200) -> list[dict]:
    """
    Returns conversation records for a session, oldest first.

    Unlike save_conversation() (which degrades silently so a write failure
    never breaks /query), this RAISES ServiceUnavailableError if Postgres
    isn't configured or the read fails. Rationale: this is called from an
    explicit GET /conversations/{session_id} request — silently returning an
    empty list would be indistinguishable from "this session genuinely has
    no history yet", which is a real, different, and legitimate case (see
    the empty-history test). A caller who can't get an answer to "what
    happened in this session" deserves to know why, not a misleading [].
    """
    pool = get_pool()
    if pool is None:
        raise ServiceUnavailableError(
            "Conversation history is unavailable because PostgreSQL is not configured."
        )

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_HISTORY_SQL, session_id, limit)
        # asyncpg Records support dict(row) directly — this keeps the
        # return type (and everything downstream in routers/conversations.py)
        # identical to the MongoDB version, which returned plain dicts too.
        return [dict(row) for row in rows]
    except ServiceUnavailableError:
        raise
    except Exception as exc:
        logger.error("Failed to read conversation history for session_id=%s: %s", session_id, exc)
        raise ServiceUnavailableError(f"Failed to read conversation history: {exc}") from exc


async def count_queries() -> int:
    """
    Total number of stored query records — powers GET /stats.

    Unlike get_conversation_history(), this degrades to 0 rather than
    raising when Postgres isn't configured. Rationale: /stats is a
    best-effort aggregate dashboard, not a request for a specific resource —
    there's no ambiguity to worry about here the way an empty conversation
    list could be mistaken for "no history yet" in Step 12. A stats page
    that shows partial numbers is more useful than one that 503s entirely
    because one of its several inputs isn't configured.
    """
    pool = get_pool()
    if pool is None:
        return 0
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(*) FROM conversations")
    except Exception as exc:
        logger.error("Failed to count queries: %s", exc)
        return 0

async def count_distinct_sessions() -> int:
    """Total number of distinct session_ids — powers GET /stats."""
    pool = get_pool()
    if pool is None:
        return 0
    try:
        async with pool.acquire() as conn:
            return await conn.fetchval("SELECT COUNT(DISTINCT session_id) FROM conversations")
    except Exception as exc:
        logger.error("Failed to count distinct sessions: %s", exc)
        return 0


    
