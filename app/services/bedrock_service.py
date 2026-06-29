"""
bedrock_service.py
------------------
Wrapper around Amazon Bedrock for:
  1. Text embedding  – Titan Text Embeddings V2  (used in ingestion + query)
  2. Text generation – Claude 3 Haiku             (used in RAG query pipeline)

All calls are made via boto3 using the EC2 instance IAM role — no static credentials.

TODO (real integration):
  - boto3 is already the real SDK; just ensure the EC2 instance role has:
      bedrock:InvokeModel on the embedding and LLM model ARNs.
  - For higher throughput consider Bedrock Provisioned Throughput.
"""

import json
import logging
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from app.config import settings

logger = logging.getLogger(__name__)


def _get_bedrock_client():
    """
    Create a Bedrock runtime client using the default credential chain
    (EC2 instance role → environment variables → ~/.aws/credentials).
    No static keys are used.
    """
    return boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Generate an embedding vector for *text* using Titan Text Embeddings V2.

    Returns a list of floats with length BEDROCK_EMBEDDING_DIMENSION (default 1024).

    Raises
    ------
    RuntimeError if the Bedrock call fails.
    """
    client = _get_bedrock_client()
    body = json.dumps({"inputText": text})

    try:
        response = client.invoke_model(
            modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        embedding: list[float] = result["embedding"]

        logger.debug(
            "embed_text: model=%s dim=%d text_preview='%.60s'",
            settings.BEDROCK_EMBEDDING_MODEL_ID,
            len(embedding),
            text,
        )
        return embedding

    except (ClientError, BotoCoreError) as exc:
        logger.error("Bedrock embed_text failed: %s", exc)
        raise RuntimeError(f"Bedrock embedding failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    """
    Send *prompt* to Claude 3 Haiku and return the model's text response.

    The prompt is expected to already contain the system instructions,
    retrieved context, and user question assembled by prompt_service.

    Raises
    ------
    RuntimeError if the Bedrock call fails.
    """
    client = _get_bedrock_client()

    # Claude 3 Messages API format
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.BEDROCK_LLM_MAX_TOKENS,
            "temperature": settings.BEDROCK_LLM_TEMPERATURE,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
    )

    try:
        response = _get_bedrock_client().invoke_model(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        answer: str = result["content"][0]["text"]

        logger.info(
            "generate_answer: model=%s tokens_used=%s",
            settings.BEDROCK_LLM_MODEL_ID,
            result.get("usage", {}).get("output_tokens", "unknown"),
        )
        return answer

    except (ClientError, BotoCoreError) as exc:
        logger.error("Bedrock generate_answer failed: %s", exc)
        raise RuntimeError(f"Bedrock generation failed: {exc}") from exc
