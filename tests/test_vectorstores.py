"""
Tests for app.vectorstores.get_vector_store() — the factory that's the
single switch point between Chroma and Pinecone.

Since get_vector_store() is lru_cache'd, each test clears the cache before
and after itself so that monkeypatching settings.vector_store for one test
can't leak a stale cached instance into the next.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.vectorstores import get_vector_store


@pytest.fixture(autouse=True)
def _reset_vector_store_cache():
    get_vector_store.cache_clear()
    yield
    get_vector_store.cache_clear()


def test_get_vector_store_returns_chroma_by_default():
    from app.vectorstores.chroma import ChromaVectorStore

    store = get_vector_store()

    assert isinstance(store, ChromaVectorStore)


def test_get_vector_store_returns_pinecone_when_configured(monkeypatch):
    """
    Proves the factory switch works without hitting the real Pinecone API.
    PineconeVectorStore.__init__ is mocked so no network call is made.
    """
    monkeypatch.setattr(settings, "vector_store", "pinecone")
    monkeypatch.setattr(settings, "pinecone_api_key", "fake-test-key")

    with patch("app.vectorstores.pinecone.PineconeVectorStore.__init__", return_value=None):
        store = get_vector_store()

    from app.vectorstores.pinecone import PineconeVectorStore
    assert isinstance(store, PineconeVectorStore)


def test_get_vector_store_raises_value_error_for_unknown_backend(monkeypatch):
    monkeypatch.setattr(settings, "vector_store", "some_unsupported_backend")

    with pytest.raises(ValueError, match="Unknown VECTOR_STORE"):
        get_vector_store()


def test_pinecone_raises_value_error_without_api_key(monkeypatch):
    """
    Confirms PineconeVectorStore fails fast with a clear error if the
    API key isn't configured — better than a cryptic network error later.
    """
    monkeypatch.setattr(settings, "vector_store", "pinecone")
    monkeypatch.setattr(settings, "pinecone_api_key", "")

    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        get_vector_store()
