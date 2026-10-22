from qdrant_client import QdrantClient

from app.services.vector_store import (
    COLLECTION_NAME,
    store_chunks,
)

def test_store_chunk()-> None:
    client = QdrantClient(":memory:")

    chunks = [
        {
            "content": "RAG retrives relevant document passages.",
            "source": "rag_basics.txt",
            "chunk_index": 0,
            "embedding": [0.1, 0.2, 0.3],
        },
        {
            "content": "Embeddings represent text numerically.",
            "source": "rag_basics.txt",
            "chunk_index": 1,
            "embedding": [0.4, 0.5, 0.6],
        }
    ]
    store_chunks(client, chunks)

    results,_ = client.scroll(
        collection_name = COLLECTION_NAME,
        limit = 10,
        with_payload = True,
        with_vectors = True
    )

    assert len(results) == 2
    assert results[0].payload['content']
    assert len(results[0].vector) == 3
    assert all(point.id for point in results)
    client.close()