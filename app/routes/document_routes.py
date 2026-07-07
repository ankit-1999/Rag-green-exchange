"""
document_routes.py
------------------
HTTP routes for the document ingestion pipeline.

Endpoints
---------
POST /documents/upload        Ingest a document from S3 into OpenSearch
GET  /documents               List all ingested document records
GET  /documents/{document_id} Get metadata for a specific document
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, status

from app.schemas.document_schema import (
    DocumentClearIndexResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentMetadata,
)
from app.services import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document from S3",
    description=(
        "Fetches the document from the provided S3 URI, splits it into chunks, "
        "generates embeddings via Bedrock Titan, and indexes into OpenSearch."
    ),
)
async def upload_document(request: DocumentUploadRequest) -> DocumentUploadResponse:
    try:
        result = document_service.ingest_document(request)
        if result.status == "failed":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.message,
            )
        return result
    except FileNotFoundError as exc:
        logger.warning("Document not found in S3: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        logger.error("Ingestion pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream service error: {exc}",
        ) from exc


@router.get(
    "",
    response_model=List[DocumentMetadata],
    summary="List all ingested documents",
)
async def list_documents() -> List[DocumentMetadata]:
    return document_service.list_documents()


@router.post(
    "/clear-index",
    response_model=DocumentClearIndexResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear all indexed chunks from OpenSearch",
)
async def clear_index() -> DocumentClearIndexResponse:
    try:
        return document_service.clear_indexed_chunks()
    except RuntimeError as exc:
        logger.error("Failed to clear OpenSearch index data: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream service error: {exc}",
        ) from exc


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get metadata for a specific document",
)
async def get_document(document_id: str) -> DocumentMetadata:
    metadata = document_service.get_document_metadata(document_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document not found: {document_id}",
        )
    return metadata
