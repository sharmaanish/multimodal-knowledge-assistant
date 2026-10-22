from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from uuid import NAMESPACE_URL, uuid5


COLLECTION_NAME = 'knowledge_documents'

def get_qdrant_client() -> QdrantClient:
    """Create a local Qdrant client that store data on disk."""

    return QdrantClient(path='./qdrant_storage')

def create_collection(
        client : QdrantClient,
        vector_size : int,
) -> None:
    """Create the collection if it does not already exist."""

    if vector_size <= 0:
        raise ValueError('vector_size must be greater than zero.')

    if client.collection_exists(COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size= vector_size,
            distance = Distance.COSINE,
        ),
    )

def store_chunks(
            client : QdrantClient,
            chunks : list[dict],
    ) -> None:
        """Store document chunks, embeddings, metadata in Qdrant."""

        if not chunks:
            raise ValueError("Chunks cannot be empty")

        vector_size = len(chunks[0]['embedding'])
        create_collection(client, vector_size)

        points = []

        for index, chunk in enumerate(chunks):
            embedding = chunk['embedding']

            if len(embedding) != vector_size:
                raise ValueError("All embeddings must have the same dimension.")

            point_id = str(

                 uuid5(
                      NAMESPACE_URL, 
                      f'{chunk['source']}:{chunk['chunk_index']}',

                    )
                 )
            points.append(
                PointStruct(
                    id = point_id,
                    vector = embedding,
                    payload = {
                        'content' : chunk['content'],
                        "source": chunk['source'],
                        "chunk_index": chunk['chunk_index']
                    }
                )
            )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points = points,
        )