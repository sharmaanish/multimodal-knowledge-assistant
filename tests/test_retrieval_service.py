from qdrant_client import QdrantClient
import pytest

from app.services.retrieval_service import search_documents
from app.services.vector_store import COLLECTION_NAME, store_chunks

def test_search_documents_return_relevant_chunks(monkeypatch) -> None:
    client = QdrantClient(":memory:")

    chunks = [
        {
            "content": 'RAG retrieves relevant document before gathering an answer',
            "source": 'documents/rag_basis.txt',
            "chunk_index" : 0,
            "embedding" : [1.0, 0.0, 0.0]
        }
        ,
        # {
        #     "content": "Photosynthesis allows plants to convert light into energy.",
        #     "source": "documents/biology.txt",
        #     "chunk_index": 0,
        #     "embedding": [0.0, 1.0, 0.0],
        # },
    ]

    try:
        store_chunks(client, chunks)

        monkeypatch.setattr(
            "app.services.retrieval_service.embed_text",
            lambda query: [1.0, 0.0, 0.0],
        )

        results = search_documents(
            "How does RAG work?",
            limit = 1,
            client = client,
        )

        assert len(results) == 1
        assert "RAG" in results[0]['content']
        assert results[0]['source'] == 'documents/rag_basis.txt'
        assert results[0]['chunk_index'] == 0
        assert isinstance(results[0]['score'], float)

    finally:
        client.close()

def test_search_documents_rejects_empty_query() -> None:

    with pytest.raises(ValueError, match="Query cannot be empty."):
        search_documents(" ")


def test_search_documents_rejects_invalid_limit() -> None:

    with pytest.raises(ValueError, match="Limit must be greater than zero."):
        search_documents("What is RAG?", limit=0)

