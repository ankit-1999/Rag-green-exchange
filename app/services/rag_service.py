import logging
from typing import Dict, List

from app.config import settings
from app.schemas.query_schema import QueryRequest, QueryResponse, QuerySource
from app.services import bedrock_service, opensearch_service, prompt_service

logger = logging.getLogger(__name__)


def _build_fallback_answer(hits: List[Dict], generation_error: Exception) -> str:
    """Return an extractive fallback answer when LLM generation is unavailable."""
    snippets: List[str] = []
    for idx, hit in enumerate(hits[:3], start=1):
        text = (hit.get("text", "") or "").strip().replace("\n", " ")
        if text:
            snippets.append(f"{idx}. {text[:280]}")

    fallback_context = "\n".join(snippets) if snippets else "No retrievable snippets available."
    return (
        "The generative model is currently unavailable, so this is an extractive fallback from retrieved context.\n"
        f"Reason: {generation_error}\n\n"
        f"Top retrieved snippets:\n{fallback_context}"
    )


def answer_question(request: QueryRequest) -> QueryResponse:
    """
    End-to-end RAG flow:
      1) Embed question
      2) Retrieve top-k chunks from OpenSearch
      3) Build prompt with retrieved context
      4) Generate answer from LLM
    """
    top_k = request.top_k or settings.OPENSEARCH_TOP_K

    question_embedding = bedrock_service.embed_text(request.question)
    hits: List[Dict] = opensearch_service.search_similar_chunks(question_embedding, top_k=top_k)

    if not hits:
        return QueryResponse(
            answer=(
                "I could not find relevant indexed context for this question yet. "
                "Please upload more related documents and try again."
            ),
            source_count=0,
            sources=[],
        )

    prompt = prompt_service.build_rag_prompt(request.question, hits)
    try:
        answer = bedrock_service.generate_answer(prompt)
    except RuntimeError as exc:
        logger.warning("LLM generation unavailable, returning fallback answer: %s", exc)
        answer = _build_fallback_answer(hits, exc)

    sources: List[QuerySource] = []
    for hit in hits:
        sources.append(
            QuerySource(
                chunk_id=hit.get("chunk_id", ""),
                document_id=hit.get("document_id", ""),
                document_name=hit.get("document_name", ""),
                document_type=hit.get("document_type", ""),
                chunk_index=int(hit.get("chunk_index", 0)),
                s3_uri=hit.get("s3_uri", ""),
                score=float(hit.get("score", 0.0)),
                snippet=(hit.get("text", "")[:280]).replace("\n", " "),
            )
        )

    logger.info("answer_question: top_k=%d sources=%d", top_k, len(sources))
    return QueryResponse(answer=answer, source_count=len(sources), sources=sources)
