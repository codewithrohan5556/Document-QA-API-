"""
Chat model client. Groq exposes an OpenAI-compatible /v1 API, so we reuse
LangChain's ChatOpenAI class rather than writing a custom integration --
same pattern the roadmap used for switching between OpenAI and Anthropic.

Every other service (rag_service, etc.) depends on `get_llm()`, never on
Groq or ChatOpenAI directly, so swapping the chat provider later means
editing this one file plus config.py.
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Singleton so we don't rebuild an HTTP client on every request."""
    if not settings.groq_api_key:
        logger.warning("GROQ_API_KEY is not set — LLM calls will fail until it is configured.")

    return ChatOpenAI(
        model=settings.groq_model,
        api_key=settings.groq_api_key or "not-set",
        base_url=settings.groq_base_url,
        max_completion_tokens=settings.max_completion_tokens,
        temperature=settings.llm_temperature,
        max_retries=settings.llm_max_retries,  # handles transient 5xx / timeouts
        request_timeout=settings.llm_timeout_seconds,
    )


def safe_invoke(prompt: str) -> str:
    """
    Direct (non-chain) call with explicit error handling for the two
    failure modes that matter most in production: rate limits and
    generic API errors. The LCEL retrieval chain (added when we build
    rag_service.py) gets the same protection for free via max_retries on
    the underlying client; this helper is for ad-hoc calls outside a chain.
    """
    llm = get_llm()
    try:
        response = llm.invoke(prompt)
        return response.content
    except RateLimitError:
        logger.error("Groq rate limit hit.")
        raise
    except APIError as e:
        logger.error(f"Groq API error: {e}")
        raise
