
import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse


def test_chat_request_accepts_valid_question() -> None:
    request = ChatRequest(question="What is RAG?")

    assert request.question == "What is RAG?"


def test_chat_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(question="")


def test_chat_response_accepts_answer_and_sources() -> None:
    response = ChatResponse(
        answer="RAG retrieves relevant documents.",
        sources=[
            {
                "source": "documents/rag_basis.txt",
                "chunk_index": 0,
            }
        ],
    )

    assert response.answer == "RAG retrieves relevant documents."
    assert response.sources[0].source == "documents/rag_basis.txt"
