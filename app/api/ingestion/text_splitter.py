from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(
        text:str,
        chunk_size : int = 500,
        chunk_overlap: int = 50,
)-> list[str]:
    """Split text into overlapping chunks for retrieval."""

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    if chunk_size <= 0 :
        raise ValueError('Chunk size must be greater than zero.')

    if chunk_overlap>= chunk_size or chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller thank chunk_size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators=['\n\n', '\n', " ", ""]
    )
    return splitter.split_text(text)
