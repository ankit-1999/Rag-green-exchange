"""
prompt_service.py
-----------------

Builds the final grounded answer prompt for GreenGrid Exchange.

The prompt combines:
- Retrieved RAG chunks for rules and explanations
- Compact marketplace API metadata
- Deterministic analytics, prediction, and recommendation results
- The user's question

The LLM must explain calculated results. It must not recalculate, alter, or
invent marketplace values.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence


SUPPORTED_ENERGY_SOURCES = ("SOLAR", "WIND", "HYDRO")

PREDICTION_INTENTS = {
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
}

RECOMMENDATION_INTENTS = {
    "seller_recommendation",
    "buyer_recommendation",
}


# ---------------------------------------------------------------------------
# Serialization and normalization
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> str:
    """Serialize values not handled by the standard JSON encoder."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _json_text(value: Any) -> str:
    """Serialize prompt context deterministically."""
    return json.dumps(
        value,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
        default=_json_default,
    )


def _normalize_question(question: str) -> str:
    normalized = (question or "").strip()
    if not normalized:
        raise ValueError("Question cannot be empty.")
    return normalized


def _needs_elaborate_response(question: str) -> bool:
    """Detect questions that explicitly request explanation or methodology."""
    normalized = question.lower()
    keywords = (
        "explain",
        "elaborate",
        "detail",
        "detailed",
        "why",
        "how",
        "method",
        "calculation",
        "compare",
        "analysis",
        "factors",
    )
    return any(keyword in normalized for keyword in keywords)


# ---------------------------------------------------------------------------
# RAG source construction
# ---------------------------------------------------------------------------


def _build_context_blocks(retrieved_chunks: Sequence[Mapping[str, Any]]) -> str:
    """Build bounded RAG context blocks with stable source identifiers."""
    blocks: List[str] = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        text = str(chunk.get("text", "") or "").strip()
        if not text:
            continue

        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    f"document_name: {chunk.get('document_name', 'unknown')}",
                    f"document_type: {chunk.get('document_type', 'unknown')}",
                    f"chunk_id: {chunk.get('chunk_id', 'unknown')}",
                    f"chunk_index: {chunk.get('chunk_index', 'unknown')}",
                    "content:",
                    text,
                ]
            )
        )

    return "\n\n".join(blocks) if blocks else "None"


# ---------------------------------------------------------------------------
# API context compaction
# ---------------------------------------------------------------------------


def _compact_api_context(api_context: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Keep deterministic results and execution metadata while removing full raw
    API record arrays from the final LLM prompt.

    Complete datasets are processed by analytics_service. Nova Micro receives
    only calculated results, record counts, filters, samples, and limitations.
    """
    if not api_context:
        return None

    compact: Dict[str, Any] = {
        "context_type": api_context.get("context_type"),
        "planner_reason": api_context.get("planner_reason"),
        "intent": api_context.get("intent", "none"),
        "is_prediction": bool(api_context.get("is_prediction", False)),
        "is_recommendation": bool(api_context.get("is_recommendation", False)),
        "historical_period": api_context.get("historical_period"),
        "forecast_period": api_context.get("forecast_period"),
        "group_by": api_context.get("group_by", []),
        "metrics": api_context.get("metrics", []),
        "filters_used": api_context.get("filters_used", []),
        "records_analyzed": api_context.get("records_analyzed", {}),
        "analytics_result": api_context.get("analytics_result", {}),
        "prediction_result": api_context.get("prediction_result"),
        "recommendation_result": api_context.get("recommendation_result"),
        "confidence": api_context.get("confidence"),
        "calculation_method": api_context.get("calculation_method"),
        "limitations": api_context.get("limitations", []),
        "missing_parameters": api_context.get("missing_parameters", []),
        "data_as_of": api_context.get("data_as_of"),
        "tool_results": [],
    }

    raw_tool_results = api_context.get("tool_results", [])
    if isinstance(raw_tool_results, list):
        for result in raw_tool_results:
            if not isinstance(result, Mapping):
                continue

            data = result.get("data", {})
            sample_records: List[Any] = []
            response_metadata: Dict[str, Any] = {}

            if isinstance(data, Mapping):
                possible_samples = data.get("sample_records", [])
                if isinstance(possible_samples, list):
                    sample_records = possible_samples[:5]

                possible_metadata = data.get("response_metadata", {})
                if isinstance(possible_metadata, Mapping):
                    response_metadata = dict(possible_metadata)

            compact["tool_results"].append(
                {
                    "tool": result.get("tool"),
                    "endpoint": result.get("endpoint"),
                    "arguments": result.get("arguments", {}),
                    "record_count": result.get("record_count", 0),
                    "pages_fetched": result.get("pages_fetched", 0),
                    "execution_status": result.get("execution_status"),
                    "error": result.get("error"),
                    "sample_records": sample_records,
                    "response_metadata": response_metadata,
                }
            )

    return compact


# ---------------------------------------------------------------------------
# Response format selection
# ---------------------------------------------------------------------------


def _response_format(
    question: str,
    api_context: Optional[Mapping[str, Any]],
) -> str:
    """Return intent-specific response instructions."""
    if not api_context:
        if _needs_elaborate_response(question):
            return (
                "Use this structure:\n"
                "1. Direct answer\n"
                "2. Explanation\n"
                "3. Applicable rules or limitations\n"
                "Keep the answer grounded in retrieved documents."
            )
        return (
            "Give a direct answer in one short paragraph. Add bullets only when "
            "they improve clarity."
        )

    intent = str(api_context.get("intent", "none") or "none")
    is_prediction = bool(api_context.get("is_prediction", False)) or intent in PREDICTION_INTENTS
    is_recommendation = (
        bool(api_context.get("is_recommendation", False))
        or intent in RECOMMENDATION_INTENTS
    )

    if is_prediction:
        return (
            "Use exactly these headings when supported by the supplied data:\n"
            "Prediction\n"
            "Key figures\n"
            "Confidence\n"
            "Data period and method\n"
            "Factors considered\n"
            "Limitations\n"
            "Lead with the predicted result. Include expected lower and upper "
            "bounds when supplied. Clearly state that the forecast is not guaranteed."
        )

    if is_recommendation:
        return (
            "Use exactly these headings when supported by the supplied data:\n"
            "Recommendation\n"
            "Why\n"
            "Supporting metrics\n"
            "Confidence\n"
            "Limitations\n"
            "Lead with the recommendation. If recommendation_result says "
            "no_strong_preference=true, explicitly say there is no strong preference. "
            "Do not present the recommendation as guaranteed or as financial advice."
        )

    if intent in {
        "historical_supply",
        "historical_demand",
        "average_selling_price",
        "demand_supply_ratio",
        "market_balance",
        "supply_stability",
        "price_volatility",
    }:
        return (
            "Use these headings:\n"
            "Finding\n"
            "Key metrics\n"
            "Period and method\n"
            "Limitations\n"
            "Lead with the main finding and keep the response concise."
        )

    return (
        "Use these headings:\n"
        "Answer\n"
        "Key metrics\n"
        "Data scope\n"
        "Lead with the direct marketplace finding."
    )


# ---------------------------------------------------------------------------
# Main prompt builder
# ---------------------------------------------------------------------------


def build_rag_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build the final deterministic prompt used by Nova Micro.

    API facts and deterministic analytics are authoritative for live numbers.
    RAG chunks are authoritative for project rules, definitions, formulas, and
    limitations. If live data and a static example conflict, live API-derived
    results take precedence.
    """
    normalized_question = _normalize_question(question)
    rag_context = _build_context_blocks(retrieved_chunks)
    compact_api_context = _compact_api_context(api_context)
    api_text = _json_text(compact_api_context) if compact_api_context else "None"
    format_instructions = _response_format(normalized_question, compact_api_context)

    return f"""
You are the grounded analytics, prediction, and recommendation assistant for
GreenGrid Exchange.

SUPPORTED ENERGY SOURCES:
- SOLAR
- WIND
- HYDRO

GROUNDING PRIORITY:
1. ANALYTICS_RESULT, PREDICTION_RESULT, and RECOMMENDATION_RESULT contain
   deterministic calculations and must be treated as authoritative.
2. Live API facts and record counts are authoritative for current marketplace
   values.
3. RAG_CONTEXT is authoritative for definitions, marketplace rules,
   methodology, and limitations.
4. If a static document example conflicts with live API-derived data, use the
   live API-derived value.

FACT DEFINITIONS:
- Current available supply means active available listing energy_kwh.
- Historical listed supply means listing energy_kwh created in the stated period.
- Realized demand means energy_kwh from completed purchases.
- Average selling price means volume-weighted realized price unless the supplied
  calculation explicitly says otherwise.
- Demand-to-supply ratio means completed demand kWh divided by listed supply kWh
  for the same scope and period.
- HYDRO is the supported source name. Do not output SMALL_HYDRO.

STRICT ANSWER RULES:
- Use only API_CONTEXT and RAG_CONTEXT.
- Do not invent, estimate, alter, round differently, or recalculate supplied
  numbers.
- Do not infer missing sources, locations, prices, quantities, dates, or status.
- Do not expose private credentials, tokens, passwords, private keys, or personal
  information.
- Do not claim that the assistant executed a purchase, listing, cancellation,
  blockchain transaction, or any state-changing operation.
- Do not call a forecast an observed fact.
- Do not call a recommendation guaranteed or provide financial advice.
- If confidence is insufficient_data, clearly say there is insufficient data for
  a reliable prediction or recommendation. Do not force a winner.
- If missing_parameters is non-empty, explain what is missing. Do not invent it.
- If API execution failed or returned partial data, mention the relevant
  limitation.
- If no matching records exist, say that no matching records were found.
- State exact historical and forecast periods when provided.
- State the calculation method when provided.
- Keep source names human-readable as Solar, Wind, and Hydro.
- Do not include document filenames or API tool names inside the prose unless the
  user explicitly asks; the client displays sources and APIs separately.

RESPONSE FORMAT:
{format_instructions}

USER QUESTION:
{normalized_question}

API_CONTEXT:
{api_text}

RAG_CONTEXT:
{rag_context}

Return only the final answer in plain text with short headings and bullets where
requested. Do not return JSON, Markdown code fences, API payloads, or hidden
reasoning.
""".strip()
