import app.main as main
from fastapi.testclient import TestClient

client = TestClient(main.app)

def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()['status'] == "ok"

def test_chat_returns_answer_and_sources(monkeypatch) -> None:
    def mock_generate_answer(question: str, limit: int =5) -> dict:
        return {
            "answer": "RAG combines retrieval with generation.",
            "sources": [
                {
                    "source":"documents/rag_basos.txt",
                    "chunk_index" : 0,
                }
            ]
        }
    monkeypatch.setattr(
        main,
        "generate_answer",
        mock_generate_answer
    )

    response = client.post(
        "/chat",
        json = {"question":"What is RAG?"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "RAG combines retrieval with generation."
    )
    assert response.json()["sources"][0]["chunk_index"] == 0

def test_chat_rejects_empty_question() -> None:
    response = client.post(
        "/chat",
        json={"question": ""},
    )

    assert response.status_code == 422

def test_chat_handles_service_validation_error(monkeypatch) -> None:
    def mock_generate_answer(question: str, limit: int = 5) -> dict:
        raise ValueError("Question cannot be empty.")

    monkeypatch.setattr(
        main,
        "generate_answer",
        mock_generate_answer,
    )

    response = client.post(
        "/chat",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."
