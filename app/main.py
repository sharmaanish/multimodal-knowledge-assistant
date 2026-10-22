from fastapi import FastAPI

from app.core.config import settings

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