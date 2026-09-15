"""
Embedding model client. NVIDIA's embedding model, accessed through
OpenRouter's OpenAI-compatible proxy -- so OpenAIEmbeddings works here the
same way ChatOpenAI works for Groq in llm_service.py.

retrieval_service and ingestion_service depend on `get_embeddings()`, never
on OpenRouter or OpenAIEmbeddings directly, so swapping the embedding
provider later means editing this one file plus config.py.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    if not settings.openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY is not set — embedding calls will fail until it is configured.")

    return OpenAIEmbeddings(
        model=settings.nvidia_embed_model,
        api_key=settings.openrouter_api_key or "not-set",
        base_url=settings.nvidia_base_url,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )
