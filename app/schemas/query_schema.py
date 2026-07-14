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
        description="documents",
    )
    total_documents: Optional[int] = None
    by_type: Dict[str, int] = Field(default_factory=dict)
    sample_document_names: List[str] = Field(default_factory=list)

    # Planner and analytics metadata.
    intent: Optional[str] = None
    is_prediction: bool = False
    is_recommendation: bool = False
    historical_period: Optional["QueryPeriod"] = None
    forecast_period: Optional["QueryPeriod"] = None
    group_by: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
    filters_used: List[Dict[str, Any]] = Field(default_factory=list)
    records_analyzed: Dict[str, int] = Field(default_factory=dict)
    analytics_result: Dict[str, Any] = Field(default_factory=dict)
    prediction_result: Optional[Dict[str, Any]] = None
    recommendation_result: Optional[Dict[str, Any]] = None
    confidence: Optional[str] = None
    limitations: List[str] = Field(default_factory=list)
    missing_parameters: List[str] = Field(default_factory=list)
    calculation_method: Optional[str] = None
    data_as_of: Optional[str] = None
    planner_reason: Optional[str] = None
    tool_results: List["QueryToolResult"] = Field(default_factory=list)


class QueryPeriod(BaseModel):
    """Date range used by analytics and forecasting queries."""

    from_date: Optional[str] = Field(default=None, alias="from")
    to_date: Optional[str] = Field(default=None, alias="to")


class QueryToolResult(BaseModel):
    """Compact result returned by each executed marketplace tool call."""

    tool: str
    data: Dict[str, Any] = Field(default_factory=dict)
    arguments: Dict[str, Any] = Field(default_factory=dict)
    record_count: int = 0
    pages_fetched: int = 0
    endpoint: Optional[str] = None
    execution_status: str = "failed"
    error: Optional[str] = None


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
