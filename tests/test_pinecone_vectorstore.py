"""
Unit tests for PineconeVectorStore (Step 19/20).

ALL Pinecone SDK calls are mocked — no real API key or network connection
required. This tests our own logic: filter translation, shape normalization
(converting Pinecone's response into the Chroma-compatible dict shape),
batch upsert splitting, delete count logic, and the missing-API-key guard.

Integration testing against the real Pinecone service requires an actual
API key and index — see the manual validation instructions in Step 20's
README section.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.core.config import settings


@pytest.fixture
def mock_pinecone_env(monkeypatch):
    monkeypatch.setattr(settings, "pinecone_api_key", "fake-key-abc")
    monkeypatch.setattr(settings, "pinecone_index_name", "test-index")
    monkeypatch.setattr(settings, "pinecone_dimension", 3)


@pytest.fixture
def pinecone_store(mock_pinecone_env):
    """
    Returns a PineconeVectorStore whose SDK internals are fully mocked.
    The fixture patches Pinecone at the class level so __init__ never
    makes a real network call.
    """
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=0)
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]
    mock_pc.Index.return_value = mock_index

    with patch("app.vectorstores.pinecone.Pinecone", return_value=mock_pc):
        from app.vectorstores.pinecone import PineconeVectorStore
        store = PineconeVectorStore()

    store._index = mock_index
    return store, mock_index


# --- Initialization ----------------------------------------------------------

def test_raises_if_api_key_not_set(monkeypatch):
    monkeypatch.setattr(settings, "pinecone_api_key", "")
    from app.vectorstores.pinecone import PineconeVectorStore
    with pytest.raises(ValueError, match="PINECONE_API_KEY"):
        PineconeVectorStore()


def test_creates_index_if_not_exists(mock_pinecone_env):
    mock_pc = MagicMock()
    mock_pc.list_indexes.return_value = []  # index doesn't exist yet
    mock_pc.Index.return_value = MagicMock(
        describe_index_stats=MagicMock(return_value=MagicMock(total_vector_count=0))
    )

    with patch("app.vectorstores.pinecone.Pinecone", return_value=mock_pc):
        from app.vectorstores.pinecone import PineconeVectorStore
        PineconeVectorStore()

    mock_pc.create_index.assert_called_once()
    call_kwargs = mock_pc.create_index.call_args.kwargs
    assert call_kwargs["name"] == "test-index"
    assert call_kwargs["dimension"] == 3
    assert call_kwargs["metric"] == "cosine"


def test_skips_creation_if_index_already_exists(mock_pinecone_env):
    mock_pc = MagicMock()
    # MagicMock(name=...) sets the mock's internal label, NOT a .name attribute.
    # We must set .name explicitly after creation so _ensure_index_exists()
    # finds it when checking `idx.name for idx in list_indexes()`.
    existing_index = MagicMock()
    existing_index.name = "test-index"
    mock_pc.list_indexes.return_value = [existing_index]
    mock_pc.Index.return_value = MagicMock(
        describe_index_stats=MagicMock(return_value=MagicMock(total_vector_count=0))
    )

    with patch("app.vectorstores.pinecone.Pinecone", return_value=mock_pc):
        from app.vectorstores.pinecone import PineconeVectorStore
        PineconeVectorStore()

    mock_pc.create_index.assert_not_called()


# --- Filter translation -------------------------------------------------------

def test_filter_translation_converts_flat_dict_to_pinecone_syntax(pinecone_store):
    store, _ = pinecone_store
    result = store._to_pinecone_filter({"document_id": "abc123"})
    assert result == {"document_id": {"$eq": "abc123"}}


def test_filter_translation_returns_none_for_empty_filter(pinecone_store):
    store, _ = pinecone_store
    assert store._to_pinecone_filter(None) is None
    assert store._to_pinecone_filter({}) is None


# --- add() -------------------------------------------------------------------

def test_add_upserts_with_text_in_metadata(pinecone_store):
    store, mock_index = pinecone_store
    store.add(
        ids=["doc1_0"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["Chunk text here."],
        metadatas=[{"document_id": "doc1", "chunk_id": 0, "page_number": 1}],
    )

    mock_index.upsert.assert_called_once()
    vectors = mock_index.upsert.call_args.kwargs["vectors"]
    assert vectors[0]["id"] == "doc1_0"
    assert vectors[0]["values"] == [0.1, 0.2, 0.3]
    assert vectors[0]["metadata"]["text"] == "Chunk text here."
    assert vectors[0]["metadata"]["document_id"] == "doc1"
    assert vectors[0]["metadata"]["page_number"] == 1


def test_add_batches_in_groups_of_100(pinecone_store):
    store, mock_index = pinecone_store
    n = 150
    store.add(
        ids=[f"id_{i}" for i in range(n)],
        embeddings=[[0.1, 0.2, 0.3]] * n,
        documents=[f"text {i}" for i in range(n)],
        metadatas=[{"document_id": "doc1", "chunk_id": i} for i in range(n)],
    )
    # 150 vectors -> 2 batches (100 + 50)
    assert mock_index.upsert.call_count == 2


# --- query() -----------------------------------------------------------------

def test_query_returns_chroma_compatible_shape(pinecone_store):
    store, mock_index = pinecone_store

    mock_match = MagicMock()
    mock_match.metadata = {"document_id": "doc1", "chunk_id": 0, "page_number": 1, "text": "Real chunk text."}
    mock_match.score = 0.92
    mock_index.query.return_value = MagicMock(matches=[mock_match])

    result = store.query(embedding=[0.1, 0.2, 0.3], top_k=5, filters={"document_id": "doc1"})

    # Shape must match what Chroma returns so retrieval_service.py works unchanged
    assert "documents" in result
    assert "metadatas" in result
    assert "distances" in result
    assert result["documents"] == [["Real chunk text."]]
    # text stripped from metadata
    assert "text" not in result["metadatas"][0][0]
    assert result["metadatas"][0][0]["document_id"] == "doc1"
    # distance = 1 - score
    assert abs(result["distances"][0][0] - (1 - 0.92)) < 1e-6


def test_query_applies_filter_translation(pinecone_store):
    store, mock_index = pinecone_store
    mock_index.query.return_value = MagicMock(matches=[])

    store.query(embedding=[0.1, 0.2, 0.3], top_k=5, filters={"document_id": "abc"})

    call_kwargs = mock_index.query.call_args.kwargs
    assert call_kwargs["filter"] == {"document_id": {"$eq": "abc"}}


# --- delete() ----------------------------------------------------------------

def test_delete_returns_zero_when_no_vectors_found(pinecone_store):
    store, mock_index = pinecone_store
    mock_index.query.return_value = MagicMock(matches=[])

    count = store.delete("doc-doesnt-exist")

    assert count == 0
    mock_index.delete.assert_not_called()


def test_delete_returns_count_and_deletes_ids(pinecone_store):
    store, mock_index = pinecone_store

    mock_matches = [MagicMock(id=f"doc1_{i}") for i in range(3)]
    mock_index.query.return_value = MagicMock(matches=mock_matches)

    count = store.delete("doc1")

    assert count == 3
    mock_index.delete.assert_called_once_with(ids=["doc1_0", "doc1_1", "doc1_2"])


# --- count_documents() -------------------------------------------------------

def test_count_documents_returns_zero_when_empty(pinecone_store):
    store, mock_index = pinecone_store
    mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=0)

    assert store.count_documents() == 0


def test_count_documents_returns_distinct_document_id_count(pinecone_store):
    store, mock_index = pinecone_store
    mock_index.describe_index_stats.return_value = MagicMock(total_vector_count=4)

    # 4 vectors but only 2 distinct document_ids
    mock_matches = [
        MagicMock(metadata={"document_id": "doc-a"}),
        MagicMock(metadata={"document_id": "doc-a"}),
        MagicMock(metadata={"document_id": "doc-b"}),
        MagicMock(metadata={"document_id": "doc-b"}),
    ]
    mock_index.query.return_value = MagicMock(matches=mock_matches)

    assert store.count_documents() == 2
