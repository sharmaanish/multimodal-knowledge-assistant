from app.services.embedding_service import embed_text

def test_embed_text_retuns_vector() -> None:
    embedding = embed_text("RAG retrives relevant document passage.")

    assert len(embedding)>0
    assert all(isinstance(value, float) for value in embedding)


def test_embed_text_rejects_empty_input() -> None:
    import pytest

    with pytest.raises(ValueError, match='Text cannot be empty.'):
        embed_text("   ")

def test_embed_text_returns_vectors() -> None:
    from app.services.embedding_service import embed_texts

    texts = [
        "RAG retrieves relevant documents.",
        "Embeddings represent text as numerical vectors.",
    ]

    embeddings = embed_texts(texts)

    assert len(embeddings) == len(texts)
    assert all(len(vector)>0 for vector in embeddings)
    assert len(embeddings[0]) == len(embeddings[1])


def test_embed_texts_rejects_empty_list()-> None:
    import pytest
    from app.services.embedding_service import embed_texts

    with pytest.raises(ValueError, match='Texts cannot be empty'):
        embed_texts([])