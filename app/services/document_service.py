"""
document_service.py
-------------------
Orchestrates the full document ingestion pipeline:

  1. Fetch raw document bytes from S3
  2. Extract plain text
  3. Split into overlapping chunks  (chunker_service)
  4. Generate embedding per chunk   (bedrock_service)
  5. Index chunk + embedding        (opensearch_service)
  6. Return summary

This is the WRITE path.  The READ path lives in rag_service.py (RAG query).

TODO (S3 integration):
  - _fetch_document_from_s3() already calls the real S3 API via boto3.
  - Ensure the EC2 instance role has s3:GetObject on the document bucket ARN.
"""

import io
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from app.config import settings
from app.schemas.document_schema import (
    DocumentClearIndexResponse,
    DocumentUploadRequest,
    DocumentUploadResponse,
    DocumentMetadata,
)
from app.services import chunker_service, bedrock_service, opensearch_service

logger = logging.getLogger(__name__)

# In-memory metadata store (replaced by DB in Phase 2)
_document_metadata_store: Dict[str, DocumentMetadata] = {}


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    Parse 's3://bucket/key/path.pdf' → ('bucket', 'key/path.pdf').

    Raises ValueError on malformed URI.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {s3_uri}")
    without_scheme = s3_uri[5:]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Malformed S3 URI: {s3_uri}")
    return parts[0], parts[1]


def _fetch_document_from_s3(s3_uri: str) -> bytes:
    """
    Download the object at *s3_uri* and return its raw bytes.

    Uses boto3 default credential chain (EC2 instance role).
    """
    bucket, key = _parse_s3_uri(s3_uri)
    s3_client = boto3.client("s3", region_name=settings.AWS_REGION)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content: bytes = response["Body"].read()
        logger.info("Fetched S3 object: bucket=%s key=%s size=%d bytes", bucket, key, len(content))
        return content
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        if error_code == "NoSuchKey":
            raise FileNotFoundError(f"S3 object not found: {s3_uri}") from exc
        raise RuntimeError(f"S3 fetch failed ({error_code}): {exc}") from exc
    except BotoCoreError as exc:
        raise RuntimeError(f"S3 fetch failed: {exc}") from exc


def _build_stable_document_id(s3_uri: str) -> str:
    """
    Build a deterministic document ID from S3 URI.

    Re-ingesting the same object URI reuses the same document_id so chunk IDs are
    overwritten instead of duplicated.
    """
    digest = hashlib.sha1(s3_uri.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _extract_text(raw_bytes: bytes, document_name: str) -> str:
    """
    Extract plain text from document bytes.

    Supported formats:
      - .txt / .md  → decode as UTF-8
      - .pdf        → TODO: integrate pdfminer.six or pypdf for production
                      For MVP, attempt UTF-8 decode; warn if it fails.

    TODO: Add pypdf / pdfminer.six for real PDF text extraction.
    """
    name_lower = document_name.lower()
    if name_lower.endswith(".pdf"):
        # TODO: Replace with real PDF extraction, e.g.:
        #   from pdfminer.high_level import extract_text_to_fp
        #   ...
        logger.warning(
            "PDF extraction is not yet implemented. "
            "Attempting raw UTF-8 decode for '%s'. "
            "Install pypdf or pdfminer.six for proper PDF support.",
            document_name,
        )
    try:
        return raw_bytes.decode("utf-8", errors="replace")
    except Exception as exc:
        raise ValueError(f"Could not decode document '{document_name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Core ingestion pipeline
# ---------------------------------------------------------------------------

def ingest_document(request: DocumentUploadRequest) -> DocumentUploadResponse:
    """
    Full ingestion pipeline for a single document.

    Steps
    -----
    1. Ensure OpenSearch index exists.
    2. Fetch raw bytes from S3.
    3. Extract text.
    4. Split into chunks.
    5. Embed each chunk via Bedrock Titan.
    6. Index chunk + embedding in OpenSearch.
    7. Persist metadata and return response.
    """
    document_id = _build_stable_document_id(request.s3_uri)
    logger.info(
        "Starting ingestion: document_id=%s name=%s s3_uri=%s",
        document_id,
        request.document_name,
        request.s3_uri,
    )

    # Step 1 — ensure index
    opensearch_service.ensure_index_exists()

    # Step 2 — fetch from S3
    raw_bytes = _fetch_document_from_s3(request.s3_uri)

    # Step 3 — extract text
    text = _extract_text(raw_bytes, request.document_name)
    if not text.strip():
        return DocumentUploadResponse(
            document_id=document_id,
            document_name=request.document_name,
            status="failed",
            chunk_count=0,
            message="Document is empty or could not be parsed.",
        )

    # Step 4 — chunk
    chunks: List[str] = chunker_service.chunk_text(text)
    if not chunks:
        return DocumentUploadResponse(
            document_id=document_id,
            document_name=request.document_name,
            status="failed",
            chunk_count=0,
            message="No chunks produced from document text.",
        )

    # Steps 5 + 6 — embed and index each chunk
    indexed_count = 0
    failed_chunks: List[int] = []

    for idx, chunk_text in enumerate(chunks):
        chunk_id = f"{document_id}_chunk_{idx}"
        try:
            embedding = bedrock_service.embed_text(chunk_text)
            opensearch_service.index_chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                document_name=request.document_name,
                document_type=request.document_type,
                chunk_index=idx,
                text=chunk_text,
                embedding=embedding,
                s3_uri=request.s3_uri,
            )
            indexed_count += 1
        except Exception as exc:
            logger.error("Failed to index chunk %d of %s: %s", idx, document_id, exc)
            failed_chunks.append(idx)

    # Step 7 — persist metadata
    metadata = DocumentMetadata(
        document_id=document_id,
        document_name=request.document_name,
        document_type=request.document_type,
        s3_uri=request.s3_uri,
        chunk_count=indexed_count,
        indexed_at=datetime.now(timezone.utc),
    )
    _document_metadata_store[document_id] = metadata

    status = "indexed"
    if failed_chunks and indexed_count == 0:
        status = "failed"
    elif failed_chunks:
        status = "partial"

    message = (
        f"Document indexed successfully. {indexed_count}/{len(chunks)} chunks stored."
        if not failed_chunks
        else f"Partial indexing: {indexed_count}/{len(chunks)} chunks stored. "
             f"Failed chunk indices: {failed_chunks}."
    )

    logger.info(
        "Ingestion complete: document_id=%s status=%s chunks=%d/%d",
        document_id,
        status,
        indexed_count,
        len(chunks),
    )

    return DocumentUploadResponse(
        document_id=document_id,
        document_name=request.document_name,
        status=status,
        chunk_count=indexed_count,
        message=message,
    )


def get_document_metadata(document_id: str) -> Optional[DocumentMetadata]:
    """Return stored metadata for a document, or None if not found."""
    return _document_metadata_store.get(document_id)


def list_documents() -> List[DocumentMetadata]:
    """Return all ingested document metadata records."""
    return list(_document_metadata_store.values())


def get_documents_summary(sample_size: int = 5) -> Dict:
    """
    Return a compact operational summary of ingested documents.

    This is used by RAG query orchestration for questions like
    "how many documents are available" where metadata is better sourced from
    application state than semantic retrieval chunks.
    """
    docs = list_documents()
    by_type: Dict[str, int] = {}
    for doc in docs:
        by_type[doc.document_type] = by_type.get(doc.document_type, 0) + 1

    sample_names: List[str] = [doc.document_name for doc in docs[: max(sample_size, 0)]]

    return {
        "total_documents": len(docs),
        "by_type": by_type,
        "sample_document_names": sample_names,
    }


def clear_indexed_chunks() -> DocumentClearIndexResponse:
    """Remove all indexed chunks from OpenSearch and clear cached metadata."""
    deleted_chunks = opensearch_service.clear_index_data()
    cleared_documents = len(_document_metadata_store)
    _document_metadata_store.clear()

    return DocumentClearIndexResponse(
        deleted_chunks=deleted_chunks,
        cleared_documents=cleared_documents,
        message="Indexed chunks and document metadata cleared successfully.",
    )
