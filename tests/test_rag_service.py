
from unittest.mock import patch

import pytest

from app.services import rag_service


def test_generate_answer_empty_question() -> None:
    with pytest.raises(ValueError, match="Question cannot be empty."):
        rag_service.generate_answer("   ")


def test_generate_answer_with_no_document_found() -> None:
    with patch(
        "app.services.rag_service.rag_graph.invoke",
        return_value={
            "answer": rag_service.INSUFFICIENT_EVIDENCE,
            "sources": [],
        },
    ) as mock_invoke:
        result = rag_service.generate_answer("What is RAG?")

    mock_invoke.assert_called_once()
    assert result["answer"] == rag_service.INSUFFICIENT_EVIDENCE
    assert result["sources"] == []


def test_generate_answer_uses_retrieved_context() -> None:
    expected_sources = [
        {
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
        }
    ]

    with patch(
        "app.services.rag_service.rag_graph.invoke",
        return_value={
            "answer": (
                "RAG retrieves relevant documents before generating an answer. "
                "[Source: documents/rag_basis.txt, Chunk: 0]"
            ),
            "sources": expected_sources,
        },
    ) as mock_invoke:
        result = rag_service.generate_answer("What is RAG?")

    mock_invoke.assert_called_once_with(
        {
            "question": "What is RAG?",
            "limit": 5,
            "retrieved_chunks": [],
            "evidence_sufficient": False,
            "answer": "",
            "sources": [],
        }
    )

    assert "RAG retrieves relevant documents" in result["answer"]
    assert result["sources"] == expected_sources


def test_generate_answer_without_citations_returns_no_sources() -> None:
    with patch(
        "app.services.rag_service.rag_graph.invoke",
        return_value={
            "answer": "RAG retrieves documents before generating an answer.",
            "sources": [],
        },
    ):
        result = rag_service.generate_answer("What is RAG?")

    assert result["sources"] == []


def test_generate_answer_ignores_unretrieved_citations() -> None:
    # The graph is responsible for validating citations against retrieved chunks.
    # The service should return the graph's validated result unchanged.
    with patch(
        "app.services.rag_service.rag_graph.invoke",
        return_value={
            "answer": (
                "RAG retrieves documents. "
                "[Source: documents/unknown.txt, Chunk: 99]"
            ),
            "sources": [],
        },
    ):
        result = rag_service.generate_answer("What is RAG?")

    assert result["sources"] == []
