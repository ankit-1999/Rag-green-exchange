"""
opensearch_service.py
---------------------
Manages all interactions with Amazon OpenSearch Service:
  - Index creation with knn_vector mapping
  - Chunk indexing (write path — ingestion pipeline)
  - kNN similarity search (read path — RAG query pipeline)

Authentication uses AWS SigV4 request signing via the EC2 instance role.
No static credentials are used.

TODO (real integration):
  - Set OPENSEARCH_ENDPOINT env var to your domain endpoint, e.g.
    https://<domain-id>.<region>.es.amazonaws.com
  - Ensure the EC2 instance role has:
      es:ESHttpGet, es:ESHttpPost, es:ESHttpPut, es:ESHttpHead
    on the OpenSearch domain ARN.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from opensearchpy.exceptions import NotFoundError, OpenSearchException

from app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def _get_client() -> OpenSearch:
    """
    Build an OpenSearch client.

    - When OPENSEARCH_USE_AWS_AUTH=true  → SigV4-signed requests (production)
    - When OPENSEARCH_USE_AWS_AUTH=false → unsigned requests (local dev / docker)
    """
    host = settings.OPENSEARCH_ENDPOINT.rstrip("/")
    # Strip scheme for the hosts list
    host_no_scheme = host.replace("https://", "").replace("http://", "")
    use_ssl = host.startswith("https")

    if settings.OPENSEARCH_USE_AWS_AUTH:
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, settings.AWS_REGION, "es")
        return OpenSearch(
            hosts=[{"host": host_no_scheme, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
    else:
        # Local / unsecured OpenSearch (docker-compose for dev)
        port = 9200
        if ":" in host_no_scheme:
            host_no_scheme, port_str = host_no_scheme.rsplit(":", 1)
            port = int(port_str)
        return OpenSearch(
            hosts=[{"host": host_no_scheme, "port": port}],
            use_ssl=use_ssl,
            verify_certs=False,
        )


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

INDEX_MAPPING = {
    "settings": {
        "index": {
            "knn": True,
            "knn.algo_param.ef_search": 100,
        }
    },
    "mappings": {
        "properties": {
            "chunk_id":      {"type": "keyword"},
            "document_id":   {"type": "keyword"},
            "document_name": {"type": "keyword"},
            "document_type": {"type": "keyword"},
            "chunk_index":   {"type": "integer"},
            "text":          {"type": "text"},
            "s3_uri":        {"type": "keyword"},
            "indexed_at":    {"type": "date"},
            "embedding": {
                "type": "knn_vector",
                "dimension": settings.BEDROCK_EMBEDDING_DIMENSION,
                "method": {
                    "name":       "hnsw",
                    "space_type": "l2",
                    "engine":     "nmslib",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
    },
}


def ensure_index_exists() -> None:
    """
    Create the knn index if it does not already exist.
    Safe to call on every application startup.
    """
    client = _get_client()
    index = settings.OPENSEARCH_INDEX_NAME
    try:
        if not client.indices.exists(index=index):
            client.indices.create(index=index, body=INDEX_MAPPING)
            logger.info("OpenSearch index created: %s", index)
        else:
            logger.debug("OpenSearch index already exists: %s", index)
    except OpenSearchException as exc:
        logger.error("ensure_index_exists failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Write path (ingestion)
# ---------------------------------------------------------------------------

def index_chunk(
    chunk_id: str,
    document_id: str,
    document_name: str,
    document_type: str,
    chunk_index: int,
    text: str,
    embedding: List[float],
    s3_uri: str,
) -> None:
    """
    Index a single document chunk with its embedding vector.

    Parameters map directly to the INDEX_MAPPING properties above.
    """
    client = _get_client()
    doc = {
        "chunk_id":      chunk_id,
        "document_id":   document_id,
        "document_name": document_name,
        "document_type": document_type,
        "chunk_index":   chunk_index,
        "text":          text,
        "s3_uri":        s3_uri,
        "indexed_at":    datetime.now(timezone.utc).isoformat(),
        "embedding":     embedding,
    }
    try:
        client.index(
            index=settings.OPENSEARCH_INDEX_NAME,
            id=chunk_id,
            body=doc,
        )
        logger.debug("Indexed chunk: %s", chunk_id)
    except OpenSearchException as exc:
        logger.error("index_chunk failed for %s: %s", chunk_id, exc)
        raise


# ---------------------------------------------------------------------------
# Read path (RAG retrieval)
# ---------------------------------------------------------------------------

def search_similar_chunks(
    query_embedding: List[float],
    top_k: int = settings.OPENSEARCH_TOP_K,
) -> List[Dict]:
    """
    Perform a kNN similarity search using the query embedding vector.

    Returns a list of hit dicts, each containing:
        chunk_id, document_name, document_type, chunk_index, text, s3_uri, score
    """
    client = _get_client()
    query_body = {
        "size": top_k,
        "_source": ["chunk_id", "document_id", "document_name", "document_type",
                    "chunk_index", "text", "s3_uri"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        },
    }
    try:
        response = client.search(
            index=settings.OPENSEARCH_INDEX_NAME,
            body=query_body,
        )
        hits = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            src["score"] = hit["_score"]
            hits.append(src)
        logger.info("search_similar_chunks: returned %d hits", len(hits))
        return hits
    except NotFoundError:
        logger.warning("OpenSearch index not found during search: %s", settings.OPENSEARCH_INDEX_NAME)
        return []
    except OpenSearchException as exc:
        logger.error("search_similar_chunks failed: %s", exc)
        raise
