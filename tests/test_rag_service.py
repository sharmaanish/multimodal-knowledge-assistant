import pytest

from app.services import rag_service

def test_generate_answer_empty_question()-> None:
    """An empty question should be rejected"""

    with pytest.raises(
        ValueError,
        match="Question cannot be empty.",
    ):
        rag_service.generate_answer(" ")

def test_generate_answer_with_no_document_found(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a fallback when retrival find no documents."""

    monkeypatch.setattr(
        rag_service,
        "search_documents",
        lambda query, limit:[]
    )

    result = rag_service.generate_answer('What is RAG?')

    assert result['answer'] == rag_service.INSUFFICIENT_EVIDENCE
    assert result['sources'] == []

def test_generate_answer_uses_retrived_conetxt(
        monkeypatch : pytest.MonkeyPatch
)-> None:
    """Generate an answer using retrieved documents chunks."""

    chunks = [
        {
            "content" : 'RAG retrieves relevant documents before generating an answer.',
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
            "score": 0.95
        }
    ]

    class MockResponse:
        content = (
            "RAG retrieves relevant document before generating an answer. "
            "[Source: documents/rag_basis.txt, Chunk: 0]"
        )
    class MockLLM:
        def invoke(self, prompt: str) -> MockResponse:
            assert 'RAG retrieves relevant documents' in prompt
            assert 'What is RAG?' in prompt
            return MockResponse()

    monkeypatch.setattr(
        rag_service,
        "search_documents",
        lambda query, limit: chunks,
    )

    monkeypatch.setattr(
        rag_service,
        "get_llm",
        lambda: MockLLM(),
    )

    result = rag_service.generate_answer('What is RAG?')

    assert "RAG retrieves relevant document before generating an answer" in result['answer']
    assert result['sources'] == [
        {
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
        }
    ]


def test_generate_answer_without_citations_returns_no_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return no source references when the answer cites none."""

    chunks = [
        {
            "content": "RAG retrieves documents before generating an answer.",
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
            "score": 0.95,
        }
    ]

    class MockResponse:
        content = "RAG retrieves relevant documents before generating an answer."

    class MockLLM:
        def invoke(self, prompt: str) -> MockResponse:
            return MockResponse()

    monkeypatch.setattr(
        rag_service,
        "search_documents",
        lambda query, limit: chunks,
    )
    monkeypatch.setattr(
        rag_service,
        "get_llm",
        lambda: MockLLM(),
    )

    result = rag_service.generate_answer("What is RAG?")

    assert result["answer"] == (
        "RAG retrieves relevant documents before generating an answer."
    )
    assert result["sources"] == []


def test_generate_answer_ignores_unretrieved_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not return citations that were not retrieved."""

    chunks = [
        {
            "content": "RAG retrieves documents before generating an answer.",
            "source": "documents/rag_basis.txt",
            "chunk_index": 0,
            "score": 0.95,
        }
    ]

    class MockResponse:
        content = (
            "RAG retrieves documents before generating an answer. "
            "[Source: documents/unknown.txt, Chunk: 99]"
        )

    class MockLLM:
        def invoke(self, prompt: str) -> MockResponse:
            return MockResponse()

    monkeypatch.setattr(
        rag_service,
        "search_documents",
        lambda query, limit: chunks,
    )
    monkeypatch.setattr(
        rag_service,
        "get_llm",
        lambda: MockLLM(),
    )

    result = rag_service.generate_answer("What is RAG?")

    assert result["sources"] == []
