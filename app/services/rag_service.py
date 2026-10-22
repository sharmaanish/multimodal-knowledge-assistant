from app.services.llm_service import get_llm
from app.services.retrieval_service import search_documents

INSUFFICIENT_EVIDENCE = (
    "I couldn't find enough information in the available documents "
    "to answer that question reliabily"
)

def generate_answer(
        question : str,
        limit : int  = 5
) -> dict:
    """Generate an answer grounded in retrived document chunks."""

    if not question.strip():
        raise ValueError('Question cannot be empty.')

    retrived_chunks = search_documents(
        query=  question,
        limit=limit
    )

    if not retrived_chunks:
        return {
            "answer" : INSUFFICIENT_EVIDENCE,
            "sources": [],
        }

    context_parts =[]
    sources = []
    seen_sources =set()

    for chunk in retrived_chunks:
        source = chunk['source']
        chunk_index = chunk['chunk_index']

        context_parts.append(
            f"Source: {source}\n"
            f"Chunk : {chunk_index}\n"
            f"Content : {chunk['content']}"
        )

        source_key = (source, chunk_index)

        if source_key not in seen_sources:
            sources.append(
                {
                    "source": source,
                    "chunk_index" : chunk_index,
                }
            )

            seen_sources.add(source_key)

        context = "\n\n".join(context_parts)

        prompt =f"""

You are a knowledge assistant that answers questions using supplied
document excerpts.

Rules:
1. Use only the information in the supplied context.
2. Do not invent facts or use outside knowledge.
3. If the context does not contain enough evidence, clearly state that
    you cannot answer reliabily from the available documents.
4. Cite supporting sources using their source path, for example
   [Source: documents/rag_basis.txt].
5. Be clear and consise.

Context:
{context}

Question:
{question}

Answer:
"""
        llm = get_llm()
        response = llm.invoke(prompt)
        answer = response.content

        if not isinstance(answer, str):
            answer = str(answer)

        return {
            "answer": answer.strip(),
            "sources": sources,
        }
