"""
FastAPI application entrypoint.

Deliberately thin: no business logic lives here. Startup/shutdown wiring
and router registration only. As routers are added (documents, query,
conversations — Steps 5+), they get `app.include_router(...)`'d here and
nowhere else.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.core.tracing import configure_tracing
from app.db.postgres import close_postgres_connection, connect_to_postgres
from app.models.responses import HealthResponse
from app.routers import conversations, documents, query
from app.utils.exceptions import DocumentQAError

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    configure_logging()
    logger = get_logger(__name__)
    logger.info("Starting %s (vector_store=%s)", settings.app_name, settings.vector_store)
    configure_tracing()
    await connect_to_postgres()

    yield

    # Shutdown
    await close_postgres_connection()
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(conversations.router)


@app.exception_handler(DocumentQAError)
async def document_qa_error_handler(request: Request, exc: DocumentQAError) -> JSONResponse:
    """
    Converts any application-specific error into a clean JSON body instead
    of a raw traceback. Routers/services just raise (e.g.
    `UnsupportedFileTypeError(...)`) and never touch HTTP concerns.
    """
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Last-resort safety net (Step 16 coverage pass surfaced this gap): any
    exception that ISN'T a DocumentQAError — a genuine bug, an unexpected
    third-party error, anything we didn't anticipate — still becomes a
    clean JSON response instead of a raw traceback reaching the client, per
    the spec's "return clean JSON errors instead of raw Python tracebacks"
    requirement. This is the officially recommended Starlette/FastAPI
    pattern for overriding the default 500 behavior.

    The real traceback is still fully logged server-side via
    logger.exception() — it's just never sent over the wire.
    """
    logger = get_logger(__name__)
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "An unexpected error occurred."})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Liveness check. Intentionally simple for Phase 1 — just confirms the
    process is up and answering requests. Does NOT check downstream
    dependencies (Chroma, PostgreSQL, Groq, etc.) yet; that's a deliberate
    "don't overcomplicate it initially" per the project spec.
    """
    return HealthResponse(status="healthy")
