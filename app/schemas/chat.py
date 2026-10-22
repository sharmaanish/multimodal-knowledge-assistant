from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Validate a user's chat request."""

    question: str = Field(
        min_length=1,
        description="The question to ask the knowledge assistant.",
        examples=['What is Retrival-Augented Generation?'],
    
    )

class SourceRefrence(BaseModel):
    """Identify a document chunk used as a source."""

    source:str
    chunk_index:int

class ChatResponse(BaseModel):
    """Define the assistant's response."""

    answer:str
    sources:list[SourceRefrence]