"""
LangSmith tracing setup.

The key mechanism to understand: LangChain's global callback manager
auto-traces every ChatOpenAI / OpenAIEmbeddings call whenever a specific set
of environment variables is present at call time. It does NOT read from our
Settings object — Settings is our own abstraction, LangChain has never heard
of it. This module's only job is bridging Settings -> os.environ, once, at
startup, so that "free" auto-tracing actually turns on.

Deliberately opt-in on TWO conditions, not one: LANGSMITH_TRACING=true AND a
real LANGSMITH_API_KEY. Enabling tracing with no key would mean every traced
call attempts (and fails) a network round-trip to LangSmith for zero
benefit — same "don't half-configure something that can't work" philosophy
used for every other *_api_key check in this project (llm_service.py,
embedding_service.py).
"""

import os

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_tracing() -> None:
    if not (settings.langsmith_tracing and settings.langsmith_api_key):
        logger.info("LangSmith tracing disabled (no LANGSMITH_API_KEY configured).")
        return

    # Both the current ("LANGSMITH_*") and legacy ("LANGCHAIN_*") env var
    # names are set. Different langchain-core / langsmith versions read one
    # or the other depending on release; setting both costs nothing and
    # avoids a silent no-op if the installed version expects the older names.
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint

    logger.info(
        "LangSmith tracing enabled (project=%s, endpoint=%s)",
        settings.langsmith_project, settings.langsmith_endpoint,
    )
