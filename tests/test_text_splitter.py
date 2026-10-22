import pytest
from app.api.ingestion.text_splitter import split_text

def test_split_text_returns()-> None:
    text = ' '.join(['RAG retrives relevant information']*50)

    chunks = split_text(text, chunk_size=100 ,chunk_overlap=20)

    assert len(chunks)>1
    assert all(len(chunk) <= 100 for chunk in chunks)

def test_split_empty_text()-> None:
    with pytest.raises(ValueError, match='Text cannot be empty'):
        split_text("")

def test_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_text("Some text", chunk_size=100, chunk_overlap=100)

