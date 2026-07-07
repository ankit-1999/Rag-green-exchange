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
from typing import Dict, List
import boto3
from botocore.exceptions import ClientError, BotoCoreError

from app.config import settings
from app.services import tool_registry

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

def embed_text(text: str) -> List[float]:
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
        embedding: List[float] = result["embedding"]

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
    Send *prompt* to the configured Bedrock chat model and return text response.

    The prompt is expected to already contain the system instructions,
    retrieved context, and user question assembled by prompt_service.

    Raises
    ------
    RuntimeError if the Bedrock call fails.
    """
    client = _get_bedrock_client()

    try:
        # Bedrock Converse API provides a common request/response shape across
        # providers (Anthropic, Nova, etc.), reducing model-specific branching.
        response = client.converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_LLM_MAX_TOKENS,
                "temperature": settings.BEDROCK_LLM_TEMPERATURE,
            },
        )
        content_blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        answer = ""
        if content_blocks:
            answer = content_blocks[0].get("text", "")

        logger.info(
            "generate_answer: model=%s tokens_used=%s",
            settings.BEDROCK_LLM_MODEL_ID,
            response.get("usage", {}).get("outputTokens", "unknown"),
        )
        return answer

    except (ClientError, BotoCoreError) as exc:
        logger.error("Bedrock generate_answer failed: %s", exc)
        raise RuntimeError(f"Bedrock generation failed: {exc}") from exc


def plan_api_calls(question: str) -> Dict:
    """
    Ask the LLM to decide whether operational API data is required before answering.

    Returns strict JSON shape:
      {
        "requires_api_data": bool,
        "reason": str,
        "tool_calls": [{"tool": str, "arguments": dict}]
      }

    If planning fails or output is malformed, default is no API call.
    """
    client = _get_bedrock_client()

    tools_text = tool_registry.build_planner_tools_text()

    planner_prompt = (
        "You are a tool planner. Decide if API data is needed before answering the user question.\n"
        "Available tools are provided as TOOL_CATALOG JSON with purpose, when_to_use, payload_schema, and response_schema.\n"
        "Only pick tool names that exist in TOOL_CATALOG.\n\n"
        f"TOOL_CATALOG:\n{tools_text}\n\n"
        "Rules:\n"
        "- If question asks for counts, ownership, price, or creation time, API data is usually required.\n"
        "- If question is conceptual/explanatory and not about live operational values, API data is not required.\n"
        "- Return only valid JSON (no markdown, no explanation outside JSON).\n\n"
        "Output JSON schema:\n"
        "{\n"
        "  \"requires_api_data\": true|false,\n"
        "  \"reason\": \"short reason\",\n"
        "  \"tool_calls\": [{\"tool\": \"tool_name_from_catalog\", \"arguments\": {}}]\n"
        "}\n\n"
        f"User question:\n{question}"
    )

    try:
        response = client.converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[{"role": "user", "content": [{"text": planner_prompt}]}],
            inferenceConfig={
                "maxTokens": 300,
                "temperature": 0.0,
            },
        )
        content_blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        planner_text = content_blocks[0].get("text", "") if content_blocks else ""
        result = json.loads(planner_text)

        requires = bool(result.get("requires_api_data", False))
        reason = str(result.get("reason", ""))
        tool_calls = result.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        return {
            "requires_api_data": requires,
            "reason": reason,
            "tool_calls": tool_calls,
        }
    except (ClientError, BotoCoreError, ValueError, json.JSONDecodeError) as exc:
        logger.warning("Bedrock plan_api_calls failed, defaulting to no API usage: %s", exc)
        return {
            "requires_api_data": False,
            "reason": "planner_unavailable",
            "tool_calls": [],
        }
