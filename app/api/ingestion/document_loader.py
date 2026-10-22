from pathlib import Path

def load_text_document(file_path:str) -> str:
    """Load a UTF-8 text document and return its content and source"""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    content = path.read_text(encoding='utf-8')

    if not content.strip():
        raise ValueError(f"Document is empty: {path}")

    return {
        "source" : str(path),
        "content": content
    }