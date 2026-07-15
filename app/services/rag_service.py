"""
rag_service.py
--------------

End-to-end orchestration for GreenGrid Exchange RAG, marketplace analytics,
prediction, and recommendation answers.

Flow:
1. Embed the user question and retrieve relevant RAG chunks.
2. Ask Nova Micro to create a validated read-only marketplace tool plan.
3. Execute approved public GET tools through marketplace_api_service.
4. Calculate deterministic analytics through analytics_service.
5. Build a compact grounded prompt.
6. Generate the final answer with Nova Micro.
7. Return RAG citations and marketplace execution metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import ValidationError

from app.config import settings
from app.schemas.query_schema import (
    QueryApiSummary,
    QueryPeriod,  # type: ignore
    QueryRequest,
    QueryResponse,
    QuerySource,
    QueryToolResult,  # type: ignore
)
from app.services import (
    analytics_service,
    bedrock_service,
    marketplace_api_service,
    opensearch_service,
    prompt_service,
    tool_registry,
)

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 4


# ---------------------------------------------------------------------------
# Marketplace tool execution
# ---------------------------------------------------------------------------


def _execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one approved public marketplace GET tool."""
    if tool_registry.get_tool_by_name(tool_name) is None:
        raise ValueError(f"Tool is not registered: {tool_name}")

    if tool_name not in {
        "get_all_listings",
        "get_active_listings",
        "get_all_purchases",
    }:
        raise ValueError(f"Tool has no read-only marketplace executor: {tool_name}")

    return marketplace_api_service.execute_tool_call(
        tool_name=tool_name,
        arguments=arguments,
    )


def _execute_plan(plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Execute validated planner calls without failing the entire query."""
    raw_calls = plan.get("tool_calls", [])
    if not isinstance(raw_calls, list):
        return []

    results: List[Dict[str, Any]] = []

    for call in raw_calls[:MAX_TOOL_CALLS]:
        if not isinstance(call, Mapping):
            continue

        tool_name = str(call.get("tool", "") or "").strip()
        arguments = call.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        try:
            result = _execute_tool_call(tool_name, arguments)
            results.append(result)
        except Exception as exc:
            logger.exception(
                "Tool execution failed before API result creation: %s",
                tool_name,
            )
            results.append(
                {
                    "tool": tool_name or "unknown",
                    "data": {
                        "records": [],
                        "sample_records": [],
                        "aggregates": {},
                        "response_metadata": {},
                    },
                    "arguments": arguments,
                    "record_count": 0,
                    "pages_fetched": 0,
                    "endpoint": None,
                    "execution_status": "failed",
                    "error": _safe_error(exc),
                }
            )

    return results


# ---------------------------------------------------------------------------
# API and analytics summary construction
# ---------------------------------------------------------------------------


def _resolve_api_summary(
    question: str,
) -> Tuple[Optional[QueryApiSummary], bool, Dict[str, Any]]:
    """
    Plan, execute, and analyze marketplace tools.

    Returns:
        (summary, api_facts_used, plan)
    """
    plan = bedrock_service.plan_api_calls(question)

    if not plan.get("requires_api_data", False):
        return None, False, plan

    tool_results = _execute_plan(plan)
    if not tool_results:
        return None, False, plan

    analysis = analytics_service.analyze_plan(
        plan=plan,
        tool_results=tool_results,
    )

    successful_or_empty = any(
        result.get("execution_status") in {"success", "partial", "empty"}
        for result in tool_results
    )

    # The new public APIs can return authoritative aggregate sections such as
    # source_breakdown, location_breakdown, supply_by_source, and
    # demand_by_source. Those aggregates count as live facts even if a response
    # contains no raw records.
    api_facts_used = any(
        result.get("execution_status") in {"success", "partial"}
        and (
            int(result.get("record_count", 0) or 0) > 0
            or _has_aggregate_data(result)
        )
        for result in tool_results
    )

    context_type = _context_type(plan)
    compact_tool_results = [
        _to_query_tool_result(result)
        for result in tool_results
    ]

    filters_used = [
        {
            "tool": result.get("tool"),
            "arguments": result.get("arguments", {}),
        }
        for result in tool_results
    ]

    summary = QueryApiSummary(
        context_type=context_type,
        planner_reason=str(plan.get("reason", "") or ""),
        intent=str(analysis.get("intent", plan.get("intent", "none"))),  # type: ignore
        is_prediction=bool(plan.get("is_prediction", False)),  # type: ignore
        is_recommendation=bool(plan.get("is_recommendation", False)),  # type: ignore
        historical_period=_to_query_period(plan.get("historical_period")),  # type: ignore
        forecast_period=_to_query_period(plan.get("forecast_period")),  # type: ignore
        group_by=list(plan.get("group_by", []) or []),  # type: ignore
        metrics=list(plan.get("metrics", []) or []),  # type: ignore
        filters_used=filters_used,  # type: ignore
        records_analyzed=dict(analysis.get("records_analyzed", {}) or {}),  # type: ignore
        analytics_result=dict(analysis.get("analytics_result", {}) or {}),  # type: ignore
        prediction_result=analysis.get("prediction_result"),  # type: ignore
        recommendation_result=analysis.get("recommendation_result"),  # type: ignore
        confidence=analysis.get("confidence"),  # type: ignore
        limitations=list(analysis.get("limitations", []) or []),  # type: ignore
        missing_parameters=list(plan.get("missing_parameters", []) or []),  # type: ignore
        calculation_method=analysis.get("calculation_method"),  # type: ignore
        data_as_of=datetime.now(timezone.utc).isoformat(),  # type: ignore
        tool_results=compact_tool_results,  # type: ignore
    )

    # Empty API responses are still useful operational context, but they do not
    # count as API facts. This lets the final answer say that no records matched.
    if successful_or_empty and not api_facts_used:
        summary.limitations = _append_unique(  # type: ignore
            summary.limitations,  # type: ignore
            "No matching marketplace records or aggregate data were available "
            "for the requested scope.",
        )

    return summary, api_facts_used, plan


def _has_aggregate_data(result: Mapping[str, Any]) -> bool:
    """Return True when a tool result contains a non-empty aggregate section."""
    raw_data = result.get("data", {})
    if not isinstance(raw_data, Mapping):
        return False

    aggregates = raw_data.get("aggregates", {})
    if not isinstance(aggregates, Mapping):
        return False

    return any(
        value not in (None, {}, [], "")
        for value in aggregates.values()
    )


def _to_query_tool_result(result: Mapping[str, Any]) -> QueryToolResult:
    """Convert a raw tool result while excluding full record arrays."""
    raw_data = result.get("data", {})
    compact_data: Dict[str, Any] = {
        "sample_records": [],
        "aggregates": {},
        "response_metadata": {},
    }

    if isinstance(raw_data, Mapping):
        samples = raw_data.get("sample_records", [])
        if isinstance(samples, list):
            compact_data["sample_records"] = samples[
                : settings.ANALYTICS_LLM_SAMPLE_RECORDS
            ]

        # Preserve the normalized aggregate sections returned by the new public
        # APIs. These are compact and useful for answer explanation, unlike the
        # full raw records array, which remains excluded.
        aggregates = raw_data.get("aggregates", {})
        if isinstance(aggregates, Mapping):
            compact_data["aggregates"] = dict(aggregates)

        metadata = raw_data.get("response_metadata", {})
        if isinstance(metadata, Mapping):
            compact_data["response_metadata"] = dict(metadata)

    return QueryToolResult(
        tool=str(result.get("tool", "unknown")),
        data=compact_data,
        arguments=dict(result.get("arguments", {}) or {}),
        record_count=int(result.get("record_count", 0) or 0),
        pages_fetched=int(result.get("pages_fetched", 0) or 0),
        endpoint=result.get("endpoint"),
        execution_status=str(result.get("execution_status", "failed")),
        error=result.get("error"),
    )


def _to_query_period(value: Any) -> Optional[QueryPeriod]:
    """Convert planner period dictionaries to the response model."""
    if not isinstance(value, Mapping):
        return None

    from_value = value.get("from")
    to_value = value.get("to")
    if not from_value and not to_value:
        return None

    return QueryPeriod(
        from_date=str(from_value) if from_value else None, # type: ignore
        to_date=str(to_value) if to_value else None, # type: ignore
    )


def _context_type(plan: Mapping[str, Any]) -> str:
    if plan.get("is_prediction", False):
        return "prediction"
    if plan.get("is_recommendation", False):
        return "recommendation"

    tool_count = len(plan.get("tool_calls", []) or [])
    intent = str(plan.get("intent", "none") or "none")
    if tool_count > 1:
        return "mixed"
    if intent not in {"none", "current_supply", "supply_mix"}:
        return "analytics"
    return "marketplace"


# ---------------------------------------------------------------------------
# RAG retrieval and source conversion
# ---------------------------------------------------------------------------


def _retrieve_chunks(question: str, top_k: int) -> List[Dict[str, Any]]:
    """Retrieve RAG chunks; allow API-backed answers if retrieval fails."""
    try:
        embedding = bedrock_service.embed_text(question)
        return opensearch_service.search_similar_chunks(
            embedding,
            top_k=top_k,
        )
    except Exception as exc:
        logger.warning("RAG retrieval unavailable: %s", exc)
        return []


def _build_sources(hits: Sequence[Mapping[str, Any]]) -> List[QuerySource]:
    sources: List[QuerySource] = []

    for hit in hits:
        try:
            sources.append(
                QuerySource(
                    chunk_id=str(hit.get("chunk_id", "") or ""),
                    document_id=str(hit.get("document_id", "") or ""),
                    document_name=str(hit.get("document_name", "") or ""),
                    document_type=str(hit.get("document_type", "") or ""),
                    chunk_index=int(hit.get("chunk_index", 0) or 0),
                    s3_uri=str(hit.get("s3_uri", "") or ""),
                    score=float(hit.get("score", 0.0) or 0.0),
                    snippet=str(hit.get("text", "") or "")[:280].replace(
                        "\n",
                        " ",
                    ),
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            logger.warning("Skipping malformed RAG source: %s", exc)

    return sources


# ---------------------------------------------------------------------------
# Answer and fallback handling
# ---------------------------------------------------------------------------


def _build_fallback_answer(
    api_summary: Optional[QueryApiSummary],
    hits: Sequence[Mapping[str, Any]],
    generation_error: Exception,
) -> str:
    """Return a useful fallback when final LLM generation is unavailable."""
    logger.warning(
        "Building fallback answer after generation failure: %s",
        generation_error,
    )

    if api_summary is not None:  # type: ignore
        if api_summary.missing_parameters:  # type: ignore
            missing = ", ".join(api_summary.missing_parameters)  # type: ignore
            return f"I need the following information to answer accurately: {missing}."

        if api_summary.confidence == "insufficient_data":  # type: ignore
            limitations = " ".join(api_summary.limitations[:2])  # type: ignore
            return (
                "There is insufficient marketplace data for a reliable answer. "
                f"{limitations}".strip()
            )

        if api_summary.prediction_result:  # type: ignore
            return (
                "The prediction was calculated, but the language-generation service "
                "is temporarily unavailable. Please retry shortly."
            )

        if api_summary.recommendation_result:  # type: ignore
            return (
                "The recommendation analysis was calculated, but the "
                "language-generation service is temporarily unavailable. "
                "Please retry shortly."
            )

        if api_summary.analytics_result:  # type: ignore
            return (
                "Marketplace analytics were calculated, but the language-generation "
                "service is temporarily unavailable. Please retry shortly."
            )

    if hits:
        return (
            "Relevant knowledge-base information was found, but the "
            "answer-generation service is temporarily unavailable. "
            "Please retry shortly."
        )

    return (
        "I could not retrieve enough information to answer this question. "
        "Please refine the question or try again shortly."
    )


def _insufficient_response(
    api_summary: Optional[QueryApiSummary],
    plan: Mapping[str, Any],
) -> str:
    missing = list(plan.get("missing_parameters", []) or [])
    if missing:
        return (
            "I need additional information before I can answer accurately: "
            + ", ".join(missing)
            + "."
        )

    if api_summary and api_summary.limitations:  # type: ignore
        return " ".join(api_summary.limitations[:2])  # type: ignore

    return "No matching marketplace or knowledge-base information was found."


def _live_data_unavailable_response(
    api_summary: Optional[QueryApiSummary],
) -> str:
    """
    Return a grounded response when a live-data question has no usable API data.

    Static RAG examples must never substitute for current or historical
    marketplace facts, predictions, or recommendations.
    """
    if api_summary and api_summary.limitations:  # type: ignore
        detail = " ".join(api_summary.limitations[:2])  # type: ignore
    else:
        detail = "The required public marketplace API returned no usable data."

    return (
        "Live marketplace data is unavailable for this question. "
        "The answer was not generated from static RAG examples. "
        f"{detail}"
    )


def _answer_mode(
    has_hits: bool,
    api_summary: Optional[QueryApiSummary],
    api_facts_used: bool,
) -> str:
    if has_hits and api_summary is not None:
        return "retrieval_plus_api"
    if api_summary is not None:
        return "api_only"
    if has_hits:
        return "retrieval_only"
    return "insufficient_data"


# ---------------------------------------------------------------------------
# Public orchestration entry point
# ---------------------------------------------------------------------------


def answer_question(request: QueryRequest) -> QueryResponse:
    """Run the complete GreenGrid grounded-answer pipeline."""
    top_k = request.top_k or settings.OPENSEARCH_TOP_K

    # Plan and API execution do not depend on successful RAG retrieval.
    api_summary, api_facts_used, plan = _resolve_api_summary(request.question)

    # New live-data safeguard: if the planner marks the question as requiring
    # live marketplace data and the APIs return neither records nor aggregates,
    # stop before RAG retrieval and answer generation. This prevents document
    # examples from being presented as live marketplace facts.
    if plan.get("requires_live_data", False) and not api_facts_used:
        return QueryResponse(
            answer=_live_data_unavailable_response(api_summary),
            source_count=0,
            sources=[],
            answer_mode="insufficient_data",
            api_facts_used=False,
            api_summary=api_summary,
        )

    hits = _retrieve_chunks(request.question, top_k)
    sources = _build_sources(hits)

    if not hits and api_summary is None:
        return QueryResponse(
            answer=_insufficient_response(None, plan),
            source_count=0,
            sources=[],
            answer_mode="insufficient_data",
            api_facts_used=False,
            api_summary=None,
        )

    prompt = prompt_service.build_rag_prompt(
        question=request.question,
        retrieved_chunks=hits,
        api_context=(
            api_summary.model_dump(by_alias=True)
            if api_summary is not None
            else None
        ),
    )

    try:
        answer = bedrock_service.generate_answer(prompt)
        if not answer.strip():
            raise RuntimeError("Bedrock returned an empty answer.")
    except (RuntimeError, ValueError) as exc:
        logger.warning("Final answer generation unavailable: %s", exc)
        answer = _build_fallback_answer(api_summary, hits, exc)

    mode = _answer_mode(bool(hits), api_summary, api_facts_used)

    logger.info(
        "answer_question: top_k=%d sources=%d mode=%s "
        "api_facts_used=%s intent=%s",
        top_k,
        len(sources),
        mode,
        api_facts_used,
        plan.get("intent", "none"),
    )

    return QueryResponse(
        answer=answer,
        source_count=len(sources),
        sources=sources,
        answer_mode=mode,
        api_facts_used=api_facts_used,
        api_summary=api_summary,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _append_unique(values: Sequence[str], new_value: str) -> List[str]:
    output = list(values)
    if new_value not in output:
        output.append(new_value)
    return output


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    return message[:500]
