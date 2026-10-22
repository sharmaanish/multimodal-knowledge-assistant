from qdrant_client import QdrantClient

from app.services.embedding_service import embed_text
from app.services.vector_store import (
    COLLECTION_NAME, get_qdrant_client
)

def search_documents(
        query:str,
        limit: int = 5,
        client: QdrantClient | None = None,
) -> list[dict]:
    """Retrive the most relevant document chunks for a query"""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if limit <= 0:
        raise ValueError('Limit must be greater than zero.')

    owns_client = client is None

    if  owns_client:
        client = get_qdrant_client()

    try:
        if not client.collection_exists(COLLECTION_NAME):
            return []

        query_embedding = embed_text(query)

        response = client.query_points(
            collection_name =COLLECTION_NAME,
            query = query_embedding,
            limit = limit,
            with_payload = True,
            with_vectors=False,
        )

        return [
            {
                "content": point.payload['content'],
                'source': point.payload['source'],
                'chunk_index': point.payload['chunk_index'],
                'score': point.score,
            }
            for point in response.points
            if point.payload is not None
        ]

    finally:
        if owns_client:
            client.closeI()