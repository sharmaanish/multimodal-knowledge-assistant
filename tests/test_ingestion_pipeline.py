from pathlib import Path
from qdrant_client import QdrantClient

from app.api.ingestion.ingestion_pipeline import (ingestion_document
,ingest_and_store_document)

from app.services.vector_store import COLLECTION_NAME
from app.api.ingestion import ingestion_pipeline
import pytest

def test_ingest_document() -> None:
    chunks = ingestion_document("documents/rag_basis.txt")

    assert len(chunks) > 0
    assert chunks[0]["content"]
    assert Path(chunks[0]["source"]) == Path("documents/rag_basis.txt")
    assert chunks[0]["chunk_index"] == 0
    assert len(chunks[0]["embedding"]) > 0
    assert all(
        isinstance(value, float)
        for value in chunks[0]["embedding"]
    )

def test_ingest_and_store_document() -> None:
    client = QdrantClient(':memory:')

    try:
        expected_chunks = ingestion_document("documents/rag_basis.txt")
        stored_count = ingest_and_store_document(
        "documents/rag_basis.txt",
        client = client,
        )

        results, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit = 100,
            with_payload=True,
            with_vectors=False,                  
                          )

        assert stored_count == len(expected_chunks)
        assert len(results) == stored_count
        assert all(point.payload['content'] for point in results)
    finally:
        client.close()

def test_ingest_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that multiple documents are ingested."""

    def mock_ingest(file_path: str) -> int:
        return 3 if file_path == "documents/doc1.txt" else 5

    monkeypatch.setattr(
        ingestion_pipeline,
        "ingest_and_store_document",
        mock_ingest,
    )

    results = ingestion_pipeline.ingest_multiple_documents(
        ["documents/doc1.txt", "documents/doc2.txt"]
    )

    assert results == {
        "documents/doc1.txt": 3,
        "documents/doc2.txt": 5,
    }


def test_ingest_multiple_documents_rejects_empty_list() -> None:
    """Verify that an empty file list is rejected."""

    with pytest.raises(
        ValueError,
        match="File paths cannot be empty.",
    ):
        ingestion_pipeline.ingest_multiple_documents([])