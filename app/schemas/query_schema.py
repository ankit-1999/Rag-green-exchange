from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for asking a grounded question over indexed chunks."""

    question: str = Field(..., min_length=3, description="Natural language user question")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of chunks to retrieve before answer generation",
    )


class QuerySource(BaseModel):
    """Source citation used to generate the answer."""

    chunk_id: str
    document_id: str
    document_name: str
    document_type: str
    chunk_index: int
    s3_uri: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """Answer plus supporting source chunks."""

    answer: str
    source_count: int
    sources: List[QuerySource]
