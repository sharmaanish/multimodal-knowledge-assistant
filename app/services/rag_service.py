
from app.graph.workflow import rag_graph

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the available documents "
    "to answer that question reliably."
)


def generate_answer(
    question: str,
    limit: int = 5,
) -> dict:
    """Generate an answer by running the LangGraph RAG workflow."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    result = rag_graph.invoke(
        {
            "question": question,
            "limit": limit,
            "retrieved_chunks": [],
            "evidence_sufficient": False,
            "answer": "",
            "sources": [],
        }
    )

    return {
        "answer": result["answer"],
        "sources": result["sources"],
    }
