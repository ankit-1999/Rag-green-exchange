from datetime import datetime
from typing import Any, Dict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inbound request models
# ---------------------------------------------------------------------------

class DocumentUploadRequest(BaseModel):
    """Metadata submitted by the caller when registering a document for ingestion."""

    document_name: str = Field(..., description="Human-readable file name, e.g. 'retirement_policy.pdf'")
    document_type: str = Field(
        ...,
        description="Category of the document: POLICY | RULE | REPORT | GUIDE",
    )
    s3_uri: str = Field(
        ...,
        description="S3 URI of the already-uploaded file, e.g. s3://bucket/key.pdf",
    )


# ---------------------------------------------------------------------------
# Outbound response models
# ---------------------------------------------------------------------------

class DocumentUploadResponse(BaseModel):
    """Returned after a document is ingested and indexed."""

    document_id: str = Field(..., description="Unique ID assigned to this document")
    document_name: str
    status: str = Field(..., description="indexed | failed | partial")
    chunk_count: int = Field(..., description="Number of chunks indexed into OpenSearch")
    message: str


class DocumentClearIndexResponse(BaseModel):
    """Returned after deleting indexed chunks from OpenSearch."""

    deleted_chunks: int
    cleared_documents: int
    message: str


# ---------------------------------------------------------------------------
# Internal / shared models
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """A single text chunk produced during ingestion and stored in OpenSearch."""

    chunk_id: str = Field(..., description="Unique ID: <document_id>_chunk_<index>")
    document_id: str
    document_name: str
    document_type: str
    chunk_index: int
    text: str
    s3_uri: str
    indexed_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentMetadata(BaseModel):
    """Lightweight metadata record stored alongside chunk index."""

    document_id: str
    document_name: str
    document_type: str
    s3_uri: str
    chunk_count: int
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = Field(default_factory=dict)
