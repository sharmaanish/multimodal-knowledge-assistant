import re
from app.services.llm_service import get_llm
from app.graph.state import RAGState
from app.services.retrieval_service import search_documents


INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the available documents "
    "to answer that question reliably."
)



def retrieve_documents(state: RAGState) -> dict:
    """Retrieve relevant document chunks for the user's question"""

    chunks = search_documents(
        query=state['question'],
        limit=state.get("limit", 5),
    )

    return {
        'retrieved_chunks' : chunks
    }

def check_evidence(state: RAGState) -> dict:
    """Check whether document chunks were retrieved."""

    chunks = state["retrieved_chunks"]

    return {
        "evidence_sufficient": len(chunks) > 0,
    }

def route_after_evidence(state: RAGState) -> str:
    """Choose the next node based on evidence availability."""

    if state["evidence_sufficient"]:
        return "generate_answer"

    return "return_fallback"

def generate_answer_node(state: RAGState) -> dict:
    """Generate an answer using the retrieved document chunks."""

    chunks = state["retrieved_chunks"]

    context = "\n\n".join(
        (
            f"Source: {chunk['source']}\n"
            f"Chunk: {chunk['chunk_index']}\n"
            f"Content: {chunk['content']}"
        )
        for chunk in chunks
    )

    prompt = f"""
You are a precise knowledge assistant.

Answer the user's question using only the provided document context.

Instructions:
- Answer the exact question directly and clearly.
- Use only information supported by the document context.
- Combine overlapping information and avoid repetition.
- Do not add unnecessary disclaimers or commentary.
- If the context does not contain enough information to answer,
  clearly state that the available documents do not provide enough evidence.
- Cite each factual claim using this exact format:
  [Source: document_path, Chunk: chunk_number]
- Use only source paths and chunk numbers shown in the context.
- Do not invent citations.

Document context:
{context}

Question:
{state["question"]}

Answer:
"""
    response = get_llm().invoke(prompt)
    answer = response.content

    if not isinstance(answer, str):
        answer = str(answer)

    # Keep only citations that refer to retrieved chunks.
    citation_pattern = r"\[Source:\s*(.*?),\s*Chunk:\s*(\d+)\]"


    retrieved_sources = {
        (
            str(chunk["source"]).replace("\\", "/"),
            str(chunk["chunk_index"]),
        )
        for chunk in chunks
    }

    # Build source references from the chunks retrieved by Qdrant.
    sources = []
    seen_sources = set()

    for chunk in chunks:
        source = str(chunk["source"]).replace("\\", "/")
        chunk_index = int(chunk["chunk_index"])
        key = (source, chunk_index)

        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(
                {
                    "source": source,
                    "chunk_index": chunk_index,
                }
            )

    return {
        "answer": answer,
        "sources": sources,
    }


def return_fallback(state: RAGState) -> dict:
    """Return a safe response when no evidence was retrieved."""

    return {
        "answer": INSUFFICIENT_EVIDENCE,
        "sources": [],
    }
