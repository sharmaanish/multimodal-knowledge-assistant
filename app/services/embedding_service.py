from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

def get_embedding_model() -> OllamaEmbeddings:
    """Create the configured local embedding model."""
    return OllamaEmbeddings(
        model = "nomic-embed-text",
        base_url=settings.ollama_base_url,
    )

def embed_text(text: str)  -> list[float]:
    """Convert a text string into a numerical embedding."""

    if not text.strip():
        raise ValueError('Text cannot be empty.')

    embedding_model = get_embedding_model()
    return embedding_model.embed_query(text)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple text strings into numerical embeddings."""

    if not texts:
        raise ValueError("Texts cannot be empty.")

    if any(not text.strip() for text in texts):
        raise ValueError('Texts cannot contain empty strings.')

    embedding_model = get_embedding_model()

    return embedding_model.embed_documents(texts)