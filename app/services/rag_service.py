import logging
import re
from typing import Dict, List, Optional, Tuple

from app.config import settings
from app.schemas.credit_schema import CreateCreditRequest, CreditTransferRequest
from app.schemas.document_schema import DocumentUploadRequest
from app.schemas.query_schema import (
    QueryApiSummary,
    QueryRequest,
    QueryResponse,
    QuerySource,
)
from app.schemas.user_schema import CreateUserRequest
from app.services import (
    bedrock_service,
    credit_service,
    document_service,
    opensearch_service,
    prompt_service,
    tool_registry,
    user_service,
)

logger = logging.getLogger(__name__)


def _extract_user_id_from_question(question: str) -> Optional[str]:
    match = re.search(r"\buser_[0-9a-f]{8}\b", question, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _is_profile_question(question: str) -> bool:
    q = question.lower()
    return "profile" in q


def _execute_tool_call(tool_name: str, arguments: Dict, question: str = "") -> Optional[Dict]:
    """Execute one approved tool call and return normalized result payload."""
    if tool_registry.get_tool_by_name(tool_name) is None:
        logger.info("Skipping tool not in registry: %s", tool_name)
        return None

    if tool_name == "create_user":
        request = CreateUserRequest(**arguments)
        user = user_service.create_user(request)
        return {"tool": tool_name, "data": user.model_dump()}

    if tool_name == "get_user":
        user_id = str(arguments.get("user_id", "")).strip()
        if not user_id:
            extracted_user_id = _extract_user_id_from_question(question)
            if extracted_user_id:
                user_id = extracted_user_id
        if not user_id:
            return None
        user = user_service.get_user(user_id)
        return {
            "tool": tool_name,
            "data": {"user": user.model_dump() if user else None},
        }

    if tool_name == "list_users":
        users = [user.model_dump() for user in user_service.list_users()]
        return {"tool": tool_name, "data": {"users": users}}

    if tool_name == "create_document":
        request = DocumentUploadRequest(**arguments)
        response = document_service.ingest_document(request)
        return {"tool": tool_name, "data": response.model_dump()}

    if tool_name == "get_documents_summary":
        data = document_service.get_documents_summary()
        return {"tool": tool_name, "data": data}

    if tool_name == "list_documents":
        limit = arguments.get("limit")
        docs = document_service.list_documents()
        if isinstance(limit, int) and limit >= 0:
            docs = docs[:limit]
        return {
            "tool": tool_name,
            "data": {"documents": [doc.model_dump() for doc in docs]},
        }

    if tool_name == "get_document":
        document_id = str(arguments.get("document_id", "")).strip()
        if not document_id:
            return None
        doc = document_service.get_document_metadata(document_id)
        return {
            "tool": tool_name,
            "data": {"document": doc.model_dump() if doc else None},
        }

    if tool_name == "create_credit":
        request = CreateCreditRequest(**arguments)
        credit = credit_service.create_credit(request)
        return {"tool": tool_name, "data": credit.model_dump()}

    if tool_name == "list_credits":
        credits = [credit.model_dump() for credit in credit_service.list_credits()]
        return {"tool": tool_name, "data": {"credits": credits}}

    if tool_name == "get_credit_details":
        credit_reference = str(arguments.get("credit_reference", "")).strip()
        if not credit_reference:
            return None
        credit = credit_service.get_credit_by_reference(credit_reference)
        return {
            "tool": tool_name,
            "data": {
                "credit_reference": credit.credit_code,
                "owner_user_id": credit.user_id,
                "credit_type": credit.credit_type.value,
                "credit_price": credit.price,
                "credit_created_at": credit.created_at,
            },
        }

    if tool_name == "list_credit_audit":
        limit = arguments.get("limit")
        audit_records = credit_service.list_credit_audit()
        if isinstance(limit, int) and limit >= 0:
            audit_records = audit_records[:limit]
        return {
            "tool": tool_name,
            "data": {
                "audit_records": [record.model_dump() for record in audit_records]
            },
        }

    if tool_name == "list_credits_created_by_user":
        user_id = str(arguments.get("user_id", "")).strip()
        if not user_id:
            return None
        credits = [
            credit.model_dump() for credit in credit_service.list_credits_created_by_user(user_id)
        ]
        return {
            "tool": tool_name,
            "data": {
                "user_id": user_id,
                "credits": credits,
            },
        }

    if tool_name == "list_credit_audit_by_credit_id":
        credit_id = str(arguments.get("credit_id", "")).strip()
        if not credit_id:
            return None
        audit_records = [
            rec.model_dump() for rec in credit_service.list_credit_audit_by_credit_id(credit_id)
        ]
        return {
            "tool": tool_name,
            "data": {
                "credit_id": credit_id,
                "audit_records": audit_records,
            },
        }

    if tool_name == "get_credit_history_timeline":
        credit_reference = str(arguments.get("credit_reference", "")).strip()
        if not credit_reference:
            return None
        timeline = credit_service.get_credit_history_timeline(credit_reference)
        return {
            "tool": tool_name,
            "data": timeline,
        }

    if tool_name == "transfer_credit":
        request = CreditTransferRequest(**arguments)
        credit = credit_service.transfer_credit(request)
        return {"tool": tool_name, "data": credit.model_dump()}

    logger.info("Skipping unsupported tool request (no executor): %s", tool_name)
    return None


def _resolve_api_summary(question: str) -> Tuple[Optional[QueryApiSummary], bool]:
    """Use LLM planner to decide and execute approved backend API tools."""
    plan = bedrock_service.plan_api_calls(question)
    requires_api_data = bool(plan.get("requires_api_data", False))
    tool_calls = plan.get("tool_calls", [])

    if _is_profile_question(question):
        extracted_user_id = _extract_user_id_from_question(question)
        if extracted_user_id:
            has_get_user = any(
                isinstance(call, dict) and str(call.get("tool", "")).strip() == "get_user"
                for call in tool_calls if isinstance(tool_calls, list)
            )
            if not has_get_user:
                if not isinstance(tool_calls, list):
                    tool_calls = []
                tool_calls.append(
                    {
                        "tool": "get_user",
                        "arguments": {"user_id": extracted_user_id},
                    }
                )
                requires_api_data = True

    if not requires_api_data:
        return None, False

    if not isinstance(tool_calls, list) or not tool_calls:
        return None, False

    tool_results: List[Dict] = []
    for tool_call in tool_calls[:4]:
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("tool", "")).strip()
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        try:
            result = _execute_tool_call(tool_name, arguments, question=question)
            if result is not None:
                tool_results.append(result)
        except Exception as exc:
            logger.warning("Tool execution failed for %s: %s", tool_name, exc)

    if not tool_results:
        return None, False

    summary = QueryApiSummary(
        context_type="mixed" if len(tool_results) > 1 else tool_results[0]["tool"],
        planner_reason=str(plan.get("reason", "")),
        tool_results=tool_results,
    )

    for result in tool_results:
        if result["tool"] == "get_documents_summary":
            data = result["data"]
            summary.total_documents = data.get("total_documents")
            summary.by_type = data.get("by_type", {})
            summary.sample_document_names = data.get("sample_document_names", [])
        elif result["tool"] == "get_credit_details":
            data = result["data"]
            summary.credit_reference = data.get("credit_reference")
            summary.owner_user_id = data.get("owner_user_id")
            summary.credit_type = data.get("credit_type")
            summary.credit_price = data.get("credit_price")
            summary.credit_created_at = data.get("credit_created_at")

    return summary, True


def _build_fallback_answer(hits: List[Dict], generation_error: Exception) -> str:
    """Return a user-friendly fallback answer when generation is unavailable."""
    return "I'm sorry, I don't have that information. Please get in touch with Ankit, who will be happy to help 🙂"


def answer_question(request: QueryRequest) -> QueryResponse:
    """
    End-to-end RAG flow:
      1) Embed question
      2) Retrieve top-k chunks from OpenSearch
      3) Optionally fetch operational API context for doc inventory questions
      4) Build prompt with API + chunk context
      5) Generate answer from LLM
    """
    top_k = request.top_k or settings.OPENSEARCH_TOP_K

    question_embedding = bedrock_service.embed_text(request.question)
    hits: List[Dict] = opensearch_service.search_similar_chunks(question_embedding, top_k=top_k)
    api_summary, api_facts_used = _resolve_api_summary(request.question)

    if not hits and not api_facts_used:
        return QueryResponse(
            answer="I'm sorry, I don't have that information. Please get in touch with Ankit, who will be happy to help 🙂",
            source_count=0,
            sources=[],
            answer_mode="retrieval_only",
            api_facts_used=False,
            api_summary=None,
        )

    prompt = prompt_service.build_rag_prompt(
        request.question,
        hits,
        api_context=api_summary.model_dump() if api_summary else None,
    )
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

    answer_mode = "retrieval_plus_api" if api_facts_used else "retrieval_only"
    logger.info(
        "answer_question: top_k=%d sources=%d api_facts_used=%s",
        top_k,
        len(sources),
        api_facts_used,
    )
    return QueryResponse(
        answer=answer,
        source_count=len(sources),
        sources=sources,
        answer_mode=answer_mode,
        api_facts_used=api_facts_used,
        api_summary=api_summary,
    )
