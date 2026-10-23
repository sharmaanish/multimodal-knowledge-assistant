from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    check_evidence,
    generate_answer_node,
    retrieve_documents,
    return_fallback,
    route_after_evidence,
)
from app.graph.state import RAGState

def build_rag_graph():
    """Build and compile the basic RAG retrival workflow."""

    graph = StateGraph(RAGState)

    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("check_evidence", check_evidence)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("return_fallback", return_fallback)
    
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", "check_evidence")
    
    graph.add_conditional_edges(
        "check_evidence",
        route_after_evidence,
        {
            "generate_answer": "generate_answer",
            "return_fallback": "return_fallback",
        },
    )

    
    graph.add_edge("generate_answer", END)
    graph.add_edge("return_fallback", END)
    return graph.compile()


rag_graph = build_rag_graph()
