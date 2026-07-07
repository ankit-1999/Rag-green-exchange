from datetime import datetime
from typing import Any, Dict, List, Optional

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


class QueryApiSummary(BaseModel):
    """Operational document facts injected into answer generation."""

    context_type: str = Field(
        default="documents",
        description="documents | credit",
    )
    total_documents: Optional[int] = None
    by_type: Dict[str, int] = Field(default_factory=dict)
    sample_document_names: List[str] = Field(default_factory=list)
    credit_reference: Optional[str] = None
    owner_user_id: Optional[str] = None
    credit_type: Optional[str] = None
    credit_price: Optional[float] = None
    credit_created_at: Optional[datetime] = None
    planner_reason: Optional[str] = None
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Answer plus supporting source chunks."""

    answer: str
    source_count: int
    sources: List[QuerySource]
    answer_mode: str = Field(
        default="retrieval_only",
        description="retrieval_only | retrieval_plus_api",
    )
    api_facts_used: bool = False
    api_summary: Optional[QueryApiSummary] = None
