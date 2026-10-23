
import re

from app.services.llm_service import get_llm
from app.services.retrieval_service import search_documents


INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the available documents "
    "to answer that question reliably."
)


def generate_answer(
    question: str,
    limit: int = 5,
) -> dict:
    """Generate an answer grounded in retrieved document chunks."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    retrieved_chunks = search_documents(
        query=question,
        limit=limit,
    )

    if not retrieved_chunks:
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "sources": [],
        }

    context_parts = []
    available_sources = []

    for chunk in retrieved_chunks:
        source = chunk["source"]
        chunk_index = chunk["chunk_index"]

        context_parts.append(
            f"Source: {source}\n"
            f"Chunk: {chunk_index}\n"
            f"Content: {chunk['content']}"
        )

        source_reference = {
            "source": source,
            "chunk_index": chunk_index,
        }

        if source_reference not in available_sources:
            available_sources.append(source_reference)

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a knowledge assistant that answers questions using
only the supplied document excerpts.

Rules:
1. Use only information supported by the supplied context.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain enough evidence, respond
   with exactly this text and nothing else:
   INSUFFICIENT_EVIDENCE
4. Cite every source used in your answer using this exact format:
   [Source: documents/example.txt, Chunk: 0]
5. Copy the source path and chunk index exactly from the context.
6. Do not cite a chunk unless it supports your answer.
7. Be clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

    response = get_llm().invoke(prompt)
    answer = response.content

    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    if answer == "INSUFFICIENT_EVIDENCE":
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "sources": [],
        }

    # Extract source paths and chunk indices cited in the answer.
    citations = re.findall(
        r"\[Source:\s*(.*?),\s*Chunk:\s*(\d+)\]",
        answer,
    )

    cited_keys = {
        (source.replace("\\", "/"), int(chunk_index))
        for source, chunk_index in citations
    }

    # Return only cited sources that were actually retrieved.
    sources = [
        source_reference
        for source_reference in available_sources
        if (
            source_reference["source"].replace("\\", "/"),
            source_reference["chunk_index"],
        ) in cited_keys
    ]

    return {
        "answer": answer,
        "sources": sources,
    }
