"""
RAG generation service.

Orchestrates: retrieve_chunks() (Step 7) -> build a grounded prompt
(app/core/prompts.py) -> call the LLM (llm_service.py, Step 3) -> shape the
response with sources.

Three entry points:
  - generate_answer()           — non-streaming, returns a complete
                                   AnswerResult.
  - stream_answer()              — streaming generator used directly by
                                   tests / anything that doesn't need
                                   persistence.
  - stream_and_record_answer()   — what routers/query.py actually calls:
                                   wraps stream_answer() to also save a
                                   ConversationRecord to PostgreSQL once
                                   streaming completes (Step 11).

LangSmith tracing (Step 14) — honest limitation, read before assuming traces
are fully unified: generate_answer() and stream_answer() are both
@traceable, and each nests the LLM call automatically (LangChain's own
callback integration traces ChatOpenAI.ainvoke/astream as a child span
within whichever @traceable function called it — no extra code needed for
that part). BUT retrieve_chunks() is called separately, in routers/query.py,
*before* stream_answer() even starts — a deliberate Step 9 design decision
so retrieval failures can still become clean JSON errors before any
streaming begins. The consequence: retrieval and generation currently show
up as two separate top-level traces per query in LangSmith, not one unified
request -> retrieval -> generation tree. Unifying them would need explicit
run-tree passing across that boundary (LangSmith supports this via
`langsmith_extra={"parent": run_tree}`), which I'm not shipping here because
I have no way to verify it against a real LangSmith backend from this
environment — noted as a documented future improvement rather than shipped
unverified.
"""

import time

from langsmith import traceable
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger
from app.core.prompts import RAG_PROMPT_TEMPLATE, build_context
from app.models.database import ConversationRecord
from app.services import conversation_service
from app.services.llm_service import get_llm
from app.services.retrieval_service import retrieve_chunks
from app.utils.exceptions import LLMError

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the document to answer that — "
    "no relevant content was found for this question."
)


class AnswerResult(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int


def _format_sources(chunks) -> list[str]:
    """
    One citation entry per retrieved chunk, in relevance order, duplicates
    removed. Uses "filename, page N" when the chunk has a real page number
    (PDFs), falling back to "filename, chunk N" when it doesn't (.txt files,
    where page_number is the -1 sentinel from ingestion_service.chunk_pages).
    """
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.page_number and chunk.page_number > 0:
            label = f"{chunk.filename}, page {chunk.page_number}"
        else:
            label = f"{chunk.filename}, chunk {chunk.chunk_id}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


@traceable(name="generate_answer", run_type="chain")
async def generate_answer(
    question: str,
    document_id: str | None = None,
    top_k: int | None = None,
) -> AnswerResult:
    chunks = retrieve_chunks(question, document_id=document_id, top_k=top_k)

    if not chunks:
        logger.info("No chunks retrieved for question — skipping LLM call.")
        return AnswerResult(answer=NO_CONTEXT_ANSWER, sources=[], chunks_used=0)

    context = build_context([chunk.model_dump() for chunk in chunks])
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        llm = get_llm()
        response = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise LLMError(f"Failed to generate an answer: {exc}") from exc

    answer = response.content
    sources = _format_sources(chunks)

    logger.info("Generated answer using %d chunks, %d sources", len(chunks), len(sources))
    return AnswerResult(answer=answer, sources=sources, chunks_used=len(chunks))


@traceable(name="stream_answer", run_type="chain")
async def stream_answer(question: str, chunks: list):
    """
    Streams the LLM's answer token by token, then a trailing sources block.

    IMPORTANT design decision: this takes already-retrieved `chunks` as an
    argument, rather than calling retrieve_chunks() itself. Retrieval can
    fail (VectorStoreError — a bad embedding call, a broken Chroma read) and
    needs to surface as a clean JSON error response. That's only possible
    BEFORE a StreamingResponse is constructed: once the first byte of a
    streamed response is sent, the HTTP status code (200) and headers are
    already committed — a mid-stream exception can no longer become a
    "502 {"error": ...}" JSON body, only a truncated response. So
    routers/query.py calls retrieve_chunks() itself, outside this generator,
    and only hands this function the results.

    A failure from the LLM *during* streaming (after tokens have already
    started arriving) has no clean way out for the same reason — so instead
    of raising, we yield a visible inline error marker and stop.
    """
    if not chunks:
        yield NO_CONTEXT_ANSWER
        return

    context = build_context([chunk.model_dump() for chunk in chunks])
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    llm = get_llm()

    try:
        async for part in llm.astream(prompt):
            if part.content:
                yield part.content
    except Exception as exc:
        logger.error("Streaming LLM call failed mid-stream: %s", exc)
        yield f"\n\n[Error: the answer could not be completed — {exc}]"
        return

    sources = _format_sources(chunks)
    if sources:
        yield "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)


async def stream_and_record_answer(
    question: str,
    chunks: list,
    document_id: str,
    session_id: str | None,
):
    """
    Wraps stream_answer() to also persist a ConversationRecord once
    streaming finishes — this is what routers/query.py actually calls.

    The client-facing stream is byte-for-byte identical to stream_answer()
    alone; persistence is a side effect that happens AFTER the last piece
    has already been yielded, using the text accumulated along the way.
    This keeps conversation storage from ever adding latency to what the
    user sees, and — per conversation_service.save_conversation()'s
    graceful-degradation — a Postgres failure here can't affect the response
    that already reached the client.

    LangSmith: we use a manual trace() context here (not @traceable) because
    this is an async generator — @traceable doesn't wrap generators cleanly.
    The trace captures the full assembled answer AFTER streaming completes,
    giving LangSmith the complete output text that astream() alone can't
    provide.
    """
    from langsmith import trace

    start = time.perf_counter()
    collected: list[str] = []

    async for piece in stream_answer(question, chunks):
        collected.append(piece)
        yield piece

    latency_ms = (time.perf_counter() - start) * 1000
    full_text = "".join(collected)
    answer_text, _, _ = full_text.partition("\n\nSources:\n")
    sources = _format_sources(chunks)

    # Log the complete assembled answer to LangSmith as a single trace —
    # this is what makes the full response visible in the UI, since the
    # streaming LLM call only shows individual tokens.
    with trace(
        name="stream_and_record_answer",
        run_type="chain",
        inputs={
            "question": question,
            "document_id": document_id,
            "session_id": session_id,
            "chunks_used": len(chunks),
        },
        metadata={"model": settings.groq_model},
    ) as run:
        run.end(
            outputs={
                "answer": answer_text,
                "sources": sources,
                "latency_ms": round(latency_ms, 2),
            }
        )

    record = ConversationRecord(
        session_id=session_id or "default",
        document_id=document_id,
        question=question,
        answer=answer_text,
        sources=sources,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in chunks],
        similarity_scores=[chunk.score for chunk in chunks],
        model=settings.groq_model,
        tokens_used=None,
        latency_ms=latency_ms,
    )
    await conversation_service.save_conversation(record)
