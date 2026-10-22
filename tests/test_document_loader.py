from pathlib import Path

import pytest

from app.api.ingestion.document_loader import load_text_document

def text_load_text_document(tmp_path:Path) -> None:
    document = tmp_path / "sample.txt"

    document.write_text(
        "RAG retrives relevant information for a question",
        encoding='utf-8'
    ) 

    result = load_text_document(str(document))

    assert result['context'] == (
        "RAG retrives relevant infromation for a question."
    )

    assert result['source'] == str(document)


def test_missing_document() ->None:
    with pytest.raises(FileNotFoundError):
                load_text_document("missing_document.txt")


def test_empty_document(tmp_path: Path) -> None:
    document = tmp_path / "empty.txt"
    document.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Document is empty"):
        load_text_document(str(document))

def test_load_text_document(tmp_path: Path) -> None:
    document = tmp_path / "sample.txt"
    document.write_text(
        "RAG retrieves relevant information for a question.",
        encoding="utf-8",
    )

    result = load_text_document(str(document))

    assert result["content"] == (
        "RAG retrieves relevant information for a question."
    )
    assert result["source"] == str(document)