"""
Centralized application configuration.

Every environment variable the app reads is declared exactly once, here.
No other module should call `os.getenv` directly — they import `settings`
from this file instead. That gives us:

  1. One place to see the full config surface of the app.
  2. Type validation and sane defaults via Pydantic (fail fast on startup
     if something required is missing or malformed, instead of failing
     deep inside a request handler later).
  3. Easy testing — settings can be overridden without touching env vars.

LLM / embedding provider note
------------------------------
Both the chat model and the embedding model are accessed through
OpenAI-compatible proxies rather than OpenAI itself:

  - Chat: Groq (https://api.groq.com/openai/v1), via langchain's ChatOpenAI.
  - Embeddings: NVIDIA's embedding model served through OpenRouter's
    OpenAI-compatible endpoint, via langchain's OpenAIEmbeddings.

This works because both providers implement the same request/response shape
as the OpenAI API — we just point `base_url` at them and pass their key
instead of an OpenAI key. It also means swapping providers later is a config
change, not a code change (same principle as the VectorStore abstraction).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Groq (chat model) ------------------------------------------------
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.2
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 60
    max_completion_tokens: int = 1024

    # --- NVIDIA embedding model (via OpenRouter's OpenAI-compatible proxy) ---
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    nvidia_base_url: str = "https://openrouter.ai/api/v1"
    nvidia_embed_model: str = "nvidia/llama-nemotron-embed-vl-1b-v2:free"

    # --- Vector store selection ---
    # "chroma" is the only implementation in Phase 1. "pinecone" arrives in Phase 2.
    vector_store: str = "chroma"
    chroma_persist_dir: str = "./data/chroma"

    # --- Pinecone (Phase 2) ---
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="document-qa", alias="PINECONE_INDEX_NAME")
    # Pinecone embedding dimension must match the embedding model output dimension.
    # NVIDIA llama-nemotron-embed-vl-1b-v2 produces 2048-dim vectors.
    pinecone_dimension: int = Field(default=2048, alias="PINECONE_DIMENSION")

    # --- PostgreSQL (wired up in Step 11, migrated from MongoDB) ---
    # Single DSN, same shape as the old MONGODB_URI — e.g.
    # postgresql://user:password@localhost:5432/document_qa
    postgres_dsn: str = Field(default="", alias="POSTGRES_DSN")

    # --- LangSmith (wired up in Step 14) ----------------------------------
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="advanced-rag", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )  # GCP US default. EU: https://eu.api.smith.langchain.com
       # APAC: https://apac.api.smith.langchain.com
       # AWS US: https://aws.api.smith.langchain.com
       # A key issued for one region 403s silently against another's endpoint.

    # --- RAG behavior — configurable, never hardcoded downstream ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5

    # --- App metadata ---
    app_name: str = "Document Q&A API"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    `lru_cache` means the .env file is read once per process, not on every
    request. FastAPI routes/services should depend on this function (or the
    `settings` instance below) rather than instantiating Settings() themselves.
    """
    return Settings()


# Convenience singleton for modules that don't need FastAPI's dependency
# injection (e.g. scripts, non-request code).
settings = get_settings()
