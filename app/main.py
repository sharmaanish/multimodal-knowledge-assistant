from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import generate_answer
from app.api.ingestion.ingestion_pipeline import (
    ingest_and_store_document
)

app = FastAPI(
    title= settings.app_name,
    version= "0.1.0",
    description= "A local knowledge assistant powered by RAG, Qdrant, Ollama."
)

@app.get("/health", tags=['Health'])
def health_check() -> dict[str, str]:
    """check wheather the API application is running."""

    return {
        "status": "ok",
        "app_name": settings.app_name
    }

@app.post('/chat', response_model= ChatResponse, tags=['Chat'])
def chat(request: ChatRequest) -> ChatResponse:
    """Answer a question using retrieved document context."""

    try:
        result = generate_answer(request.question)
        return ChatResponse(**result)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail = str(exc)
        ) from exc

@app.post('/ingest', tags = ['Ingestion'])
def ingest_document()-> dict[str, object]:
    file_path = Path("documents/rag_basis.txt")

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {file_path}"
        )
    try: 
        chunk_stored = ingest_and_store_document(str(file_path))

        return {
            "message" : str(file_path),
            "file": str(file_path),
            "chunks_stored" : chunk_stored
        }
    except ValueError as exc:
        raise HTTPException(
            status_code= 400,
            detail=str(exc),
        ) from exc

@app.post("/upload", tags=['Ingestion'])
def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    """Upload a text document and store its chunks in Qdrant"""

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail = "A filename is required."
        )

    original_name = Path(file.filename).name

    if Path(original_name).suffix.lower() != ".txt":
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported."
        )

    content = file.file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail = "The uploaded file cannot be empty."
        )
    
    try :
        text_content = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code = 400,
            detail = "The uploaded file must use UTF-8 encoding."
        ) from exc

    upload_dir = Path("documents/uploads")
    upload_dir.mkdir(parents = True, exist_ok=True)

    saved_path = upload_dir/ f"{uuid4().hex}_{original_name}"
    saved_path.write_text(text_content, encoding="utf-8")

    try:
        chunks_stored = ingest_and_store_document(str(saved_path))

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    finally:
        file.file.close()

    return {
        "message": "Document uploaded and ingested successfully.",
        "filename": original_name,
        "chunks_stored": chunks_stored,

    }