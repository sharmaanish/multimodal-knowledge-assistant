from langchain_ollama import ChatOllama

from app.core.config import settings

def get_llm() -> ChatOllama:
    """Create and return configured local chat model."""
    return ChatOllama(
        model = settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )