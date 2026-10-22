from app.api.ingestion.document_loader import load_text_document
from app.api.ingestion.text_splitter import split_text
from app.services.embedding_service import embed_texts
from qdrant_client import QdrantClient

from app.services.vector_store import get_qdrant_client, store_chunks


def ingestion_document(file_path: str) -> list[dict]:
    """Load a document, split it into chunks, and preserve source metadata"""

    document = load_text_document(file_path)
    chunks = split_text(document['content'])
    embeddings = embed_texts(chunks)

    return [
        {
            "content": chunk,
            "source": document['source'],
            "chunk_index": index,
            "embedding":embedding
        }
        for index, (chunk, embedding)  in enumerate(
            zip(chunks, embeddings, strict= True)
        )
    ]

def ingest_and_store_document(
        file_path: str,
        client : QdrantClient | None = None,
) ->  int:
    """Ingest a document and store its chunk in Qdrant"""

    chunks = ingestion_document(file_path)

    owns_client = client is None
    
    if client is None:
        client = get_qdrant_client()

    try:
        store_chunks(client, chunks)
    finally:
        if owns_client:
            client.close()

    return len(chunks)

def ingest_multiple_documents(
    file_paths: list[str],
) -> dict[str, int]:
    """Ingest multiple documents into the shared Qdrant collection."""

    if not file_paths:
        raise ValueError("File paths cannot be empty.")

    results: dict[str, int] = {}

    for file_path in file_paths:
        chunks_stored = ingest_and_store_document(file_path)
        results[file_path] = chunks_stored

    return results
