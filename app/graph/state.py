
from typing import TypedDict


class RAGState(TypedDict):
    """Store the data passed between LangGraph workflow nodes."""

    question: str
    limit: int
    retrieved_chunks: list[dict]
    evidence_sufficient: bool
    answer: str
    sources: list[dict]
