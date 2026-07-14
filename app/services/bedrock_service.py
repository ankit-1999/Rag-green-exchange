"""
bedrock_service.py
------------------

Amazon Bedrock integration for GreenGrid Exchange.

Responsibilities:
1. Generate document embeddings with Amazon Titan Text Embeddings V2.
2. Generate grounded answers with Amazon Nova Micro.
3. Plan approved read-only marketplace API calls for factual, analytical,
   predictive, and recommendation questions.

The planner may select only tools registered in tool_registry. It never executes
API calls and must never select write, authenticated, or blockchain operations.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.services import tool_registry

logger = logging.getLogger(__name__)

SUPPORTED_ENERGY_SOURCES = {"SOLAR", "WIND", "HYDRO"}
ENERGY_SOURCE_ALIASES = {
    "solar": "SOLAR",
    "solar energy": "SOLAR",
    "solar power": "SOLAR",
    "wind": "WIND",
    "wind energy": "WIND",
    "wind power": "WIND",
    "hydro": "HYDRO",
    "hydropower": "HYDRO",
    "hydro energy": "HYDRO",
    "hydro power": "HYDRO",
    # Accept old wording but always emit HYDRO.
    "small hydro": "HYDRO",
    "small-hydro": "HYDRO",
    "small_hydro": "HYDRO",
}

LISTING_STATUSES = {"active", "sold", "expired", "cancelled"}
PURCHASE_STATUSES = {
    "active",
    "pending",
    "completed",
    "consumed",
    "cancelled",
    "failed",
}
SORT_FIELDS = {"price_per_kwh", "energy_kwh", "created_at", "expires_at"}
SORT_ORDERS = {"asc", "desc"}

MARKETPLACE_TOOLS = {
    "get_all_listings",
    "get_active_listings",
    "get_all_purchases",
}

ANALYTICS_INTENTS = {
    "none",
    "current_supply",
    "supply_mix",
    "historical_supply",
    "historical_demand",
    "average_selling_price",
    "demand_supply_ratio",
    "market_balance",
    "supply_stability",
    "price_volatility",
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
    "seller_recommendation",
    "buyer_recommendation",
}

PREDICTION_INTENTS = {
    "demand_prediction",
    "price_prediction",
    "shortage_prediction",
}

RECOMMENDATION_INTENTS = {
    "seller_recommendation",
    "buyer_recommendation",
}

ALLOWED_GROUP_BY = {"energy_source", "location", "day", "week", "month"}
ALLOWED_METRICS = {
    "energy_kwh",
    "price_per_kwh",
    "total_price",
    "listing_count",
    "purchase_count",
    "demand_supply_ratio",
    "market_balance",
    "price_volatility",
    "supply_stability",
}


def _get_bedrock_client():
    """Create a Bedrock Runtime client using the default AWS credential chain."""
    return boto3.client("bedrock-runtime", region_name=settings.AWS_REGION)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embed_text(text: str) -> List[float]:
    """Generate a normalized Titan Text Embeddings V2 vector."""
    normalized_text = (text or "").strip()
    if not normalized_text:
        raise ValueError("Cannot embed empty text.")

    client = _get_bedrock_client()
    body = json.dumps(
        {
            "inputText": normalized_text,
            "dimensions": settings.BEDROCK_EMBEDDING_DIMENSION,
            "normalize": True,
        }
    )

    try:
        response = client.invoke_model(
            modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        embedding = result.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError("Bedrock returned no valid embedding vector.")

        if len(embedding) != settings.BEDROCK_EMBEDDING_DIMENSION:
            logger.warning(
                "Embedding dimension mismatch: expected=%d actual=%d",
                settings.BEDROCK_EMBEDDING_DIMENSION,
                len(embedding),
            )

        return [float(value) for value in embedding]

    except (ClientError, BotoCoreError) as exc:
        logger.exception("Bedrock embedding request failed")
        raise RuntimeError(f"Bedrock embedding failed: {exc}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.exception("Invalid Bedrock embedding response")
        raise RuntimeError(f"Invalid Bedrock embedding response: {exc}") from exc


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------


def generate_answer(prompt: str) -> str:
    """Generate the final grounded answer with the configured Nova model."""
    normalized_prompt = (prompt or "").strip()
    if not normalized_prompt:
        raise ValueError("Prompt cannot be empty.")

    client = _get_bedrock_client()

    try:
        response = client.converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": normalized_prompt}],
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
        answer = "\n".join(
            block["text"].strip()
            for block in content_blocks
            if isinstance(block, Mapping)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ).strip()

        logger.info(
            "generate_answer: model=%s input_tokens=%s output_tokens=%s",
            settings.BEDROCK_LLM_MODEL_ID,
            response.get("usage", {}).get("inputTokens", "unknown"),
            response.get("usage", {}).get("outputTokens", "unknown"),
        )
        return answer

    except (ClientError, BotoCoreError) as exc:
        logger.exception("Bedrock answer generation failed")
        raise RuntimeError(f"Bedrock generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Date context
# ---------------------------------------------------------------------------


def _date_context(today: Optional[date] = None) -> Dict[str, str]:
    current = today or datetime.now(timezone.utc).date()
    this_week_start = current - timedelta(days=current.weekday())
    this_week_end = this_week_start + timedelta(days=6)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = this_week_start - timedelta(days=1)
    this_month_start = current.replace(day=1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)

    if this_month_start.month == 12:
        next_month_start = date(this_month_start.year + 1, 1, 1)
    else:
        next_month_start = date(
            this_month_start.year,
            this_month_start.month + 1,
            1,
        )

    if next_month_start.month == 12:
        month_after_next = date(next_month_start.year + 1, 1, 1)
    else:
        month_after_next = date(
            next_month_start.year,
            next_month_start.month + 1,
            1,
        )
    next_month_end = month_after_next - timedelta(days=1)

    return {
        "today": current.isoformat(),
        "this_week_start": this_week_start.isoformat(),
        "this_week_end": this_week_end.isoformat(),
        "last_week_start": last_week_start.isoformat(),
        "last_week_end": last_week_end.isoformat(),
        "this_month_start": this_month_start.isoformat(),
        "this_month_to_date_end": current.isoformat(),
        "last_month_start": last_month_start.isoformat(),
        "last_month_end": last_month_end.isoformat(),
        "next_month_start": next_month_start.isoformat(),
        "next_month_end": next_month_end.isoformat(),
        "rolling_7_days_start": (current - timedelta(days=6)).isoformat(),
        "rolling_28_days_start": (current - timedelta(days=27)).isoformat(),
        "rolling_90_days_start": (current - timedelta(days=89)).isoformat(),
        "rolling_180_days_start": (current - timedelta(days=179)).isoformat(),
    }


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------


def _build_planner_prompt(
    question: str,
    tools_text: str,
    dates: Mapping[str, str],
) -> str:
    return f"""
You are the read-only API tool planner for GreenGrid Exchange.

GreenGrid supports only these renewable sources:
- SOLAR
- WIND
- HYDRO

Your task is to decide whether current or historical marketplace API data is
required before answering the user's question. Select only tools found in
TOOL_CATALOG.

STRICT SAFETY RULES:
- Select only read-only GET tools from TOOL_CATALOG.
- Never select a POST, PATCH, DELETE, ingestion, purchase, cancellation,
  authenticated-user, or blockchain operation.
- Never invent a tool, endpoint, filter, record, value, forecast, or date.
- Conceptual questions that can be answered from RAG documents do not require
  API data.
- Questions involving current listings, availability, quantity, location,
  price, demand, supply, trends, analytics, predictions, or recommendations
  require API data.

TOOL ROUTING:
1. get_active_listings
   Use for current availability, current active supply, source supply mix,
   active listings by location, price/quantity constrained listings, current
   recommendation candidates, and current inventory for shortage prediction.

2. get_all_listings
   Use for historical listed supply, listing status, asking-price history,
   supply trends, supply stability, demand-to-supply calculations, and
   prediction inputs.

3. get_all_purchases
   Use for completed demand, completed purchase volume, realized selling price,
   historical demand, price volatility, prediction inputs, and recommendation
   inputs. Use status=completed unless the question explicitly requires another
   supported status.

MULTI-TOOL RULES:
- Demand-to-supply ratio requires get_all_listings and get_all_purchases with
  the same period.
- Market balance or historical shortage/surplus requires get_all_listings and
  get_all_purchases with the same source, location, and period.
- Demand prediction requires historical get_all_purchases and get_all_listings.
- Price prediction requires historical get_all_purchases and current
  get_active_listings.
- Location/source shortage prediction requires historical get_all_listings,
  historical get_all_purchases, and current get_active_listings.
- Seller recommendation requires get_active_listings, recent get_all_listings,
  and recent get_all_purchases.
- Buyer recommendation requires active candidate listings and historical
  completed purchases.

FILTER RULES:
- Normalize Solar to SOLAR, Wind to WIND, and any hydro wording to HYDRO.
- Use created_from/created_to only for listing tools.
- Use completed_from/completed_to only for purchase tools.
- Use ISO-8601 dates.
- Use skip=0 and limit=100.
- Do not apply an energy_source filter when comparing all sources.
- Use location only when the user asks about a location.
- If the question says a selected period but provides no period, put
  "date_range" in missing_parameters and do not invent dates.
- For a prediction with no history period, use rolling_180_days_start through
  today as historical API filters.
- "Next month" is a forecast period, not a historical API filter.
- For "Should I list Solar or Wind credits this week?", use the previous
  28 days as supporting history and current active listings. Do not filter to a
  single source because both sources must be compared.

INTENT VALUES:
{json.dumps(sorted(ANALYTICS_INTENTS), ensure_ascii=True)}

DATE_CONTEXT:
{json.dumps(dict(dates), ensure_ascii=True, indent=2)}

TOOL_CATALOG:
{tools_text}

Return only one valid JSON object and no Markdown.

OUTPUT SCHEMA:
{{
  "requires_api_data": true,
  "reason": "short reason",
  "intent": "one allowed intent",
  "is_prediction": false,
  "is_recommendation": false,
  "historical_period": {{"from": null, "to": null}},
  "forecast_period": {{"from": null, "to": null}},
  "group_by": [],
  "metrics": [],
  "missing_parameters": [],
  "tool_calls": [
    {{"tool": "tool_name_from_catalog", "arguments": {{}}}}
  ]
}}

If API data is not required, return requires_api_data=false and tool_calls=[].

User question:
{question}
""".strip()


# ---------------------------------------------------------------------------
# Planner execution
# ---------------------------------------------------------------------------


def plan_api_calls(question: str) -> Dict[str, Any]:
    """Create and validate a read-only marketplace API plan."""
    normalized_question = (question or "").strip()
    if not normalized_question:
        return _empty_plan("empty_question")

    tools_text = tool_registry.build_planner_tools_text()
    dates = _date_context()
    prompt = _build_planner_prompt(normalized_question, tools_text, dates)
    client = _get_bedrock_client()

    logger.info(
        "plan_api_calls: question=%r tools=%s",
        normalized_question,
        tool_registry.get_allowed_tool_names(),
    )

    try:
        response = client.converse(
            modelId=settings.BEDROCK_LLM_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            inferenceConfig={
                "maxTokens": settings.BEDROCK_PLANNER_MAX_TOKENS,
                "temperature": settings.BEDROCK_PLANNER_TEMPERATURE,
            },
        )
        content_blocks = (
            response.get("output", {})
            .get("message", {})
            .get("content", [])
        )
        planner_text = "\n".join(
            block["text"].strip()
            for block in content_blocks
            if isinstance(block, Mapping)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        )
        raw = _parse_json_object(planner_text)
        plan = _normalize_plan(raw)

        logger.info(
            "plan_api_calls: requires_api_data=%s intent=%s calls=%s missing=%s",
            plan["requires_api_data"],
            plan["intent"],
            plan["tool_calls"],
            plan["missing_parameters"],
        )
        return plan

    except (ClientError, BotoCoreError) as exc:
        logger.exception("Bedrock planner invocation failed")
        return _empty_plan(f"planner_bedrock_error:{type(exc).__name__}")
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Planner returned invalid output: %s", exc)
        return _empty_plan("planner_invalid_output")


# ---------------------------------------------------------------------------
# Planner validation
# ---------------------------------------------------------------------------


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Planner returned empty output.")

    cleaned = text.strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])

    if not isinstance(value, dict):
        raise ValueError("Planner output must be a JSON object.")
    return value


def _normalize_plan(raw: Mapping[str, Any]) -> Dict[str, Any]:
    intent = str(raw.get("intent", "none") or "none").strip().lower()
    if intent not in ANALYTICS_INTENTS:
        intent = "none"

    tool_calls = _normalize_tool_calls(raw.get("tool_calls"))
    requires_api = bool(raw.get("requires_api_data", False)) or bool(tool_calls)
    if requires_api and not tool_calls:
        requires_api = False

    prediction = bool(raw.get("is_prediction", False)) or intent in PREDICTION_INTENTS
    recommendation = (
        bool(raw.get("is_recommendation", False))
        or intent in RECOMMENDATION_INTENTS
    )

    return {
        "requires_api_data": requires_api,
        "reason": str(raw.get("reason", "") or "").strip(),
        "intent": intent,
        "is_prediction": prediction,
        "is_recommendation": recommendation,
        "historical_period": _normalize_period(raw.get("historical_period")),
        "forecast_period": _normalize_period(raw.get("forecast_period")),
        "group_by": _normalize_string_list(raw.get("group_by"), ALLOWED_GROUP_BY),
        "metrics": _normalize_string_list(raw.get("metrics"), ALLOWED_METRICS),
        "missing_parameters": _normalize_missing(raw.get("missing_parameters")),
        "tool_calls": tool_calls,
    }


def _normalize_tool_calls(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    allowed_tools = set(tool_registry.get_allowed_tool_names()) & MARKETPLACE_TOOLS
    output: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for raw_call in value:
        if not isinstance(raw_call, Mapping):
            continue
        tool = str(raw_call.get("tool", "") or "").strip()
        if tool not in allowed_tools:
            logger.warning("Dropping unavailable planner tool: %r", tool)
            continue
        arguments = _normalize_tool_arguments(tool, raw_call.get("arguments"))
        fingerprint = json.dumps(
            {"tool": tool, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=True,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        output.append({"tool": tool, "arguments": arguments})

    return output[:4]


def _normalize_tool_arguments(tool: str, value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    definition = tool_registry.get_tool_by_name(tool) or {}
    schema = definition.get("payload_schema", {})
    allowed = set(schema.keys()) if isinstance(schema, Mapping) else set()
    cleaned: Dict[str, Any] = {}

    for key, item in raw.items():
        if key not in allowed or item is None or item == "":
            continue

        if key == "energy_source":
            source = _normalize_source(item)
            if source:
                cleaned[key] = source
        elif key == "location":
            if isinstance(item, str) and item.strip():
                cleaned[key] = item.strip()[:200]
        elif key == "status":
            status = str(item).strip().lower()
            valid = PURCHASE_STATUSES if tool == "get_all_purchases" else LISTING_STATUSES
            if status in valid:
                cleaned[key] = status
        elif key in {"created_from", "created_to", "completed_from", "completed_to"}:
            parsed = _normalize_iso_value(item)
            if parsed:
                cleaned[key] = parsed
        elif key in {"min_price_per_kwh", "max_price_per_kwh"}:
            number = _positive_float(item)
            if number is not None:
                cleaned[key] = number
        elif key == "min_energy_kwh":
            number = _positive_int(item)
            if number is not None:
                cleaned[key] = number
        elif key == "sort_by":
            sort_field = str(item).strip().lower()
            if sort_field in SORT_FIELDS:
                cleaned[key] = sort_field
        elif key == "sort_order":
            sort_order = str(item).strip().lower()
            if sort_order in SORT_ORDERS:
                cleaned[key] = sort_order
        elif key == "skip":
            number = _non_negative_int(item)
            if number is not None:
                cleaned[key] = number
        elif key == "limit":
            number = _positive_int(item)
            if number is not None:
                cleaned[key] = min(number, settings.MARKETPLACE_API_PAGE_SIZE)

    cleaned.setdefault("skip", 0)
    cleaned.setdefault("limit", settings.MARKETPLACE_API_PAGE_SIZE)
    if tool == "get_all_purchases":
        cleaned.setdefault("status", "completed")
    return cleaned


def _normalize_source(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    direct = raw.upper().replace("-", "_").replace(" ", "_")
    if direct in SUPPORTED_ENERGY_SOURCES:
        return direct
    return ENERGY_SOURCE_ALIASES.get(raw.lower())


def _normalize_iso_value(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        date.fromisoformat(candidate)
        return candidate
    except ValueError:
        pass
    try:
        datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        return candidate
    except ValueError:
        return None


def _positive_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _non_negative_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _normalize_period(value: Any) -> Dict[str, Optional[str]]:
    if not isinstance(value, Mapping):
        return {"from": None, "to": None}
    return {
        "from": _normalize_iso_value(value.get("from")),
        "to": _normalize_iso_value(value.get("to")),
    }


def _normalize_string_list(value: Any, allowed: Set[str]) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        normalized = str(item or "").strip().lower()
        if normalized in allowed and normalized not in output:
            output.append(normalized)
    return output


def _normalize_missing(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        normalized = str(item or "").strip()
        if normalized and normalized not in output:
            output.append(normalized)
    return output


def _empty_plan(reason: str) -> Dict[str, Any]:
    return {
        "requires_api_data": False,
        "reason": reason,
        "intent": "none",
        "is_prediction": False,
        "is_recommendation": False,
        "historical_period": {"from": None, "to": None},
        "forecast_period": {"from": None, "to": None},
        "group_by": [],
        "metrics": [],
        "missing_parameters": [],
        "tool_calls": [],
    }
