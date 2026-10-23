
from unittest.mock import patch

from app.graph.workflow import rag_graph


def test_graph_returns_fallback_when_no_chunks_found() -> None:
    with patch(
        "app.graph.nodes.search_documents",
        return_value=[],
    ):
        result = rag_graph.invoke(
            {
                "question": "What is RAG?",
                "retrieved_chunks": [],
                "evidence_sufficient": False,
                "answer": "",
                "sources": [],
            }
        )

    assert result["evidence_sufficient"] is False
    assert "couldn't find enough information" in result["answer"]
    assert result["sources"] == []


def test_graph_generates_answer_when_chunks_found() -> None:
    chunks = [
        {
            "content": "RAG combines retrieval with generation.",
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
            "score": 0.9,
        }
    ]

    mock_response = type(
        "MockResponse",
        (),
        {
            "content": (
                "RAG combines retrieval with generation. "
                "[Source: documents/rag_basis.txt, Chunk: 0]"
            )
        },
    )()

    with (
        patch(
            "app.graph.nodes.search_documents",
            return_value=chunks,
        ),
        patch(
            "app.graph.nodes.get_llm",
        ) as mock_get_llm,
    ):
        mock_get_llm.return_value.invoke.return_value = mock_response

        result = rag_graph.invoke(
            {
                "question": "What is RAG?",
                "retrieved_chunks": [],
                "evidence_sufficient": False,
                "answer": "",
                "sources": [],
            }
        )

    assert result["evidence_sufficient"] is True
    assert "RAG combines retrieval with generation" in result["answer"]
    assert result["sources"] == [
        {
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
        }
    ]
